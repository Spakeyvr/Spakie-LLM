"""MLX SFT fine-tuning loop."""

from __future__ import annotations

import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from model.transformer_mlx import SpakieGPTMLX
from runtime.mlx_backend import (
    MLXRuntimeSettings,
    clip_grads,
    save_meta_json,
)
from training.dataset_mlx import ResumableBatchSamplerMLX, stack_batch
from training.pretrain_mlx import DualAdamW, _accum_grads, _build_microbatch_step
from training.prefetch_mlx import BatchPrefetcher


def _save_sft_checkpoint(base_path: str, model: SpakieGPTMLX, meta: dict) -> None:
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    flat = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(base_path, flat, metadata={})
    save_meta_json(base_path + ".meta.json", meta)


def finetune_mlx(
    model: SpakieGPTMLX,
    train_dataset,
    val_dataset,
    config: SpakieConfig,
    runtime: MLXRuntimeSettings,
    best_checkpoint_name: str = "sft_best.safetensors",
    interrupt_checkpoint_name: str = "sft_interrupt.safetensors",
    *,
    use_compile: bool = True,
    use_prefetch: bool = True,
) -> float:
    if runtime.dtype != mx.float32:
        model.set_dtype(runtime.dtype)
    model.train()

    optimizer = DualAdamW(
        learning_rate=config.sft_lr,
        weight_decay=config.sft_weight_decay,
        betas=(0.9, 0.95),
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_sampler = ResumableBatchSamplerMLX(
        len(train_dataset), config.sft_batch_size, drop_last=True, seed=0
    )
    val_sampler = ResumableBatchSamplerMLX(
        len(val_dataset), config.sft_batch_size, drop_last=False, seed=1
    )
    steps_per_epoch = len(train_dataset) // config.sft_batch_size
    total_steps = max(1, (steps_per_epoch * config.sft_epochs) // config.sft_grad_accum_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False
    accum_scale = 1.0 / config.sft_grad_accum_steps
    microbatch_step = _build_microbatch_step(model, accum_scale, compile_step=use_compile)

    try:
        for epoch in range(config.sft_epochs):
            model.train()
            epoch_loss = 0.0
            n_micro = 0
            accum_grads = None
            micro_in_step = 0

            pbar = tqdm(
                range(steps_per_epoch),
                desc=f"SFT[mlx] Epoch {epoch + 1}/{config.sft_epochs}",
            )
            prefetcher = (
                BatchPrefetcher(train_dataset, train_sampler) if use_prefetch else None
            )
            train_iter = None if prefetcher is not None else iter(train_sampler)

            def next_batch():
                nonlocal train_iter
                if prefetcher is not None:
                    return next(prefetcher)
                try:
                    batch_indices = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_sampler)
                    batch_indices = next(train_iter)
                return stack_batch(train_dataset, batch_indices)

            try:
                for _ in pbar:
                    x_np, y_np = next_batch()
                    x = mx.array(x_np)
                    y = mx.array(y_np)

                    # Loss-fn has accum_scale baked in.
                    loss, grads = microbatch_step(x, y)
                    accum_grads = _accum_grads(accum_grads, grads)
                    # Flush lazy graph per microbatch (prevents unbounded trace growth).
                    mx.eval(accum_grads)
                    # loss is pre-scaled by accum_scale; recover unscaled for display.
                    epoch_loss += float(loss.item()) * config.sft_grad_accum_steps
                    n_micro += 1
                    micro_in_step += 1

                    if micro_in_step == config.sft_grad_accum_steps:
                        clipped, _ = clip_grads(accum_grads, config.sft_grad_clip)

                        progress = global_step / max(total_steps, 1)
                        lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (
                            1 + math.cos(math.pi * progress)
                        )
                        optimizer.set_lr(lr)
                        optimizer.update(model, clipped)
                        mx.eval(model.parameters(), optimizer.decay.state, optimizer.nodecay.state)

                        accum_grads = None
                        micro_in_step = 0
                        global_step += 1

                    pbar.set_postfix(loss=f"{epoch_loss / max(n_micro, 1):.4f}")
            finally:
                if prefetcher is not None:
                    prefetcher.close()
                pbar.close()

            # Validation
            model.eval()
            val_loss = 0.0
            val_count = 0
            val_iter = iter(val_sampler)
            for _ in range(max(1, len(val_dataset) // config.sft_batch_size)):
                try:
                    batch_indices = next(val_iter)
                except StopIteration:
                    break
                x_np, y_np = stack_batch(val_dataset, batch_indices)
                x = mx.array(x_np)
                y = mx.array(y_np)
                _, loss, _ = model(x, y)
                mx.eval(loss)
                val_loss += float(loss.item())
                val_count += 1
            val_loss = val_loss / max(val_count, 1)
            model.train()

            print(
                f"Epoch {epoch + 1} | train_loss {epoch_loss / max(n_micro, 1):.4f} | "
                f"val_loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(config.checkpoint_dir, best_checkpoint_name)
                _save_sft_checkpoint(
                    ckpt_path,
                    model,
                    meta={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "preset_name": config.preset_name,
                    },
                )
                print(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.sft_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, interrupt_checkpoint_name)
        _save_sft_checkpoint(
            interrupt_path,
            model,
            meta={"step": global_step, "best_val_loss": best_val_loss, "preset_name": config.preset_name},
        )
        print(f"\nInterrupted during fine-tuning. Saved checkpoint to {interrupt_path}")

    if interrupted:
        print(f"SFT stopped early. Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"SFT complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss
