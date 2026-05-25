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
from training.mlx_profile import MLXProfile, now
from training.optimizers_mlx import configure_mlx_optimizer
from training.pretrain_mlx import _accum_grads, _arrays_to_mx, _build_microbatch_step
from training.prefetch_mlx import BatchPrefetcher


def _save_sft_checkpoint(base_path: str, model: SpakieGPTMLX, meta: dict) -> None:
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    flat = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(base_path, flat, metadata={})
    save_meta_json(base_path + ".meta.json", meta)


def _evaluate_sft_loss(model: SpakieGPTMLX, val_dataset, batch_size: int) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    try:
        for start in range(0, len(val_dataset), batch_size):
            stop = min(start + batch_size, len(val_dataset))
            x_np, y_np = stack_batch(val_dataset, range(start, stop))
            valid_tokens = int((y_np != -100).sum())
            if valid_tokens == 0:
                continue

            x = mx.array(x_np)
            y = mx.array(y_np)
            _, loss, _ = model(x, y, return_cache=False)
            mx.eval(loss)
            total_loss += float(loss.item()) * valid_tokens
            total_tokens += valid_tokens
    finally:
        if was_training:
            model.train()

    return total_loss / max(total_tokens, 1)


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
    profile: bool = False,
) -> float:
    if runtime.dtype != mx.float32:
        model.set_dtype(runtime.dtype)
    model.train()

    optimizer = configure_mlx_optimizer(
        model,
        config,
        kind=config.sft_optimizer,
        learning_rate=config.sft_lr,
        weight_decay=config.sft_weight_decay,
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_sampler = ResumableBatchSamplerMLX(
        len(train_dataset), config.sft_batch_size, drop_last=True, seed=0
    )
    steps_per_epoch = len(train_dataset) // config.sft_batch_size
    total_steps = max(1, (steps_per_epoch * config.sft_epochs) // config.sft_grad_accum_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False
    accum_scale = 1.0 / config.sft_grad_accum_steps
    microbatch_step = _build_microbatch_step(model, accum_scale, compile_step=use_compile)
    profiler = MLXProfile(enabled=profile)

    try:
        for epoch in range(config.sft_epochs):
            model.train()
            epoch_loss = 0.0
            n_micro = 0
            accum_grads = None
            accum_loss_lazy = mx.array(0.0, dtype=mx.float32)
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
                    if profiler.enabled:
                        batch_start = now()
                        x_np, y_np = next_batch()
                        profiler.add("batch_fetch", now() - batch_start)
                    else:
                        x_np, y_np = next_batch()
                    x, y = _arrays_to_mx(x_np, y_np, profiler)

                    # Loss-fn has accum_scale baked in.
                    if profiler.enabled:
                        step_start = now()
                        loss, grads = microbatch_step(x, y)
                        mx.eval(loss, grads)
                        profiler.add("forward_backward", now() - step_start)
                    else:
                        loss, grads = microbatch_step(x, y)
                    accum_grads = _accum_grads(accum_grads, grads)
                    # loss is pre-scaled by accum_scale; sum lazily and only
                    # materialize at end of the optimizer step (or for postfix
                    # every accum_steps microbatches). Avoids forcing a CPU↔GPU
                    # sync between microbatches.
                    accum_loss_lazy = accum_loss_lazy + loss.astype(mx.float32)
                    n_micro += 1
                    micro_in_step += 1

                    if micro_in_step == config.sft_grad_accum_steps:
                        if profiler.enabled:
                            opt_start = now()
                            clipped, _ = clip_grads(accum_grads, config.sft_grad_clip)
                        else:
                            clipped, _ = clip_grads(accum_grads, config.sft_grad_clip)

                        progress = global_step / max(total_steps, 1)
                        lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (
                            1 + math.cos(math.pi * progress)
                        )
                        optimizer.set_lr(lr)
                        optimizer.update(model, clipped)
                        mx.eval(model.parameters(), optimizer.state_trees())
                        if profiler.enabled:
                            profiler.add("opt_step", now() - opt_start)

                        # accum_loss_lazy = sum of accum_steps pre-scaled losses
                        # = mean per-microbatch unscaled loss. Times accum_steps
                        # gives the sum of unscaled losses for this optimizer step,
                        # matching the previous per-microbatch accumulation.
                        step_sum_loss = float(accum_loss_lazy.item()) * config.sft_grad_accum_steps
                        epoch_loss += step_sum_loss
                        accum_grads = None
                        accum_loss_lazy = mx.array(0.0, dtype=mx.float32)
                        micro_in_step = 0
                        global_step += 1
                        pbar.set_postfix(loss=f"{epoch_loss / max(n_micro, 1):.4f}")
            finally:
                if prefetcher is not None:
                    prefetcher.close()
                pbar.close()

            # Validation
            if profiler.enabled:
                eval_start = now()
                val_loss = _evaluate_sft_loss(model, val_dataset, config.sft_batch_size)
                profiler.add("eval", now() - eval_start)
            else:
                val_loss = _evaluate_sft_loss(model, val_dataset, config.sft_batch_size)

            print(
                f"Epoch {epoch + 1} | train_loss {epoch_loss / max(n_micro, 1):.4f} | "
                f"val_loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(config.checkpoint_dir, best_checkpoint_name)
                if profiler.enabled:
                    checkpoint_start = now()
                    _save_sft_checkpoint(
                        ckpt_path,
                        model,
                        meta={
                            "epoch": epoch + 1,
                            "val_loss": val_loss,
                            "preset_name": config.preset_name,
                            "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                            "optimizer_warning": "fallback_not_recommended"
                            if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                            else "",
                            "muon_verified": config.muon_verified,
                        },
                    )
                    profiler.add("checkpoint", now() - checkpoint_start)
                else:
                    _save_sft_checkpoint(
                        ckpt_path,
                        model,
                        meta={
                            "epoch": epoch + 1,
                            "val_loss": val_loss,
                            "preset_name": config.preset_name,
                            "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                            "optimizer_warning": "fallback_not_recommended"
                            if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                            else "",
                            "muon_verified": config.muon_verified,
                        },
                    )
                print(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.sft_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    if profiler.enabled:
                        print(profiler.format_report(window_label=f"epoch {epoch + 1}"))
                        profiler.reset()
                    break
            if profiler.enabled:
                print(profiler.format_report(window_label=f"epoch {epoch + 1}"))
                profiler.reset()
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, interrupt_checkpoint_name)
        if profiler.enabled:
            checkpoint_start = now()
            _save_sft_checkpoint(
                interrupt_path,
                model,
                meta={
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                    "preset_name": config.preset_name,
                    "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                    "optimizer_warning": "fallback_not_recommended"
                    if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                    else "",
                    "muon_verified": config.muon_verified,
                },
            )
            profiler.add("checkpoint", now() - checkpoint_start)
        else:
            _save_sft_checkpoint(
                interrupt_path,
                model,
                meta={
                    "step": global_step,
                    "best_val_loss": best_val_loss,
                    "preset_name": config.preset_name,
                    "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                    "optimizer_warning": "fallback_not_recommended"
                    if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                    else "",
                    "muon_verified": config.muon_verified,
                },
            )
        print(f"\nInterrupted during fine-tuning. Saved checkpoint to {interrupt_path}")

    if interrupted:
        print(f"SFT stopped early. Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"SFT complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss
