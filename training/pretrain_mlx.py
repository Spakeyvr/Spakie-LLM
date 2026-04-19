"""MLX pretraining loop: cosine LR, grad accumulation, resumable checkpoints."""

from __future__ import annotations

import math
import os
import random
import signal
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from model.transformer_mlx import SpakieGPTMLX
from runtime.mlx_backend import (
    MLXRuntimeSettings,
    clip_grads,
    load_meta_json,
    load_safetensors,
    save_meta_json,
    save_safetensors,
    flatten_tree,
)
from training.dataset_mlx import (
    PretrainDatasetMLX,
    ResumableBatchSamplerMLX,
    stack_batch,
)


def _split_decay_grads(grads):
    """Return (decay_subset, nodecay_subset) of a grad tree. Split by ndim >= 2."""
    flat = tree_flatten(grads)
    decay = [(k, v) for k, v in flat if v is not None and v.ndim >= 2]
    nodecay = [(k, v) for k, v in flat if v is not None and v.ndim < 2]
    return tree_unflatten(decay), tree_unflatten(nodecay)


class DualAdamW:
    """Two AdamW optimizers: one with weight decay on >=2D params, one without on <2D params.

    Mirrors the PyTorch param-group pattern from training/pretrain.py.
    """

    def __init__(self, learning_rate: float, weight_decay: float, betas: tuple[float, float]):
        self._betas = betas
        self._weight_decay = weight_decay
        self.decay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
        )
        self.nodecay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=0.0, betas=betas
        )

    @property
    def learning_rate(self):
        return self.decay.learning_rate

    def set_lr(self, lr: float) -> None:
        self.decay.learning_rate = lr
        self.nodecay.learning_rate = lr

    def update(self, model, grads) -> None:
        decay_grads, nodecay_grads = _split_decay_grads(grads)
        self.decay.update(model, decay_grads)
        self.nodecay.update(model, nodecay_grads)

    def state_trees(self) -> dict:
        return {"decay": self.decay.state, "nodecay": self.nodecay.state}

    def load_state_trees(self, state: dict) -> None:
        # Assigning to .state rehydrates the internal moments; MLX optimizers accept this.
        self.decay.state = state["decay"]
        self.nodecay.state = state["nodecay"]


def get_lr(step: int, config: SpakieConfig) -> float:
    """Cosine schedule with linear warmup, decaying to 10% of peak."""
    min_lr = config.pretrain_lr * 0.1
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / config.pretrain_warmup_steps
    if step >= config.pretrain_max_steps:
        return min_lr
    progress = (step - config.pretrain_warmup_steps) / (
        config.pretrain_max_steps - config.pretrain_warmup_steps
    )
    return min_lr + 0.5 * (config.pretrain_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _build_loss_and_grad(model: SpakieGPTMLX, accum_scale: float):
    """Build value_and_grad with the grad-accum scale baked into the loss.

    Pre-scaling here means grads come out already divided by accum_steps — no
    Python-side tree_map needed between microbatches, which keeps the lazy
    graph small.
    """
    def loss_fn(model, x, y):
        _, loss, _ = model(x, y)
        return loss * accum_scale

    return nn.value_and_grad(model, loss_fn)


def _accum_grads(acc, new):
    if acc is None:
        return new
    flat_acc = dict(tree_flatten(acc))
    flat_new = dict(tree_flatten(new))
    summed = {k: flat_acc[k] + flat_new[k] for k in flat_acc}
    return tree_unflatten(list(summed.items()))


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "mlx_seed_state": None,  # MLX doesn't expose global RNG state; we rely on explicit seeds.
    }


def _restore_rng(state: dict | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])


def save_training_checkpoint_mlx(
    base_path: str,
    *,
    model: SpakieGPTMLX,
    optimizer: DualAdamW,
    config: SpakieConfig,
    global_step: int,
    tokens_processed: int,
    best_val_loss: float,
    val_loss: float | None,
    elapsed_time: float,
    train_sampler: ResumableBatchSamplerMLX,
) -> None:
    """Write a .safetensors weights file plus a sibling .meta.json."""
    weights = {"model": dict(tree_flatten(model.parameters()))}
    weights["optimizer"] = {
        "decay": dict(tree_flatten(optimizer.decay.state)),
        "nodecay": dict(tree_flatten(optimizer.nodecay.state)),
    }
    # Flatten all array leaves into a single dict for safetensors.
    flat: dict[str, mx.array] = {}
    for k, v in weights["model"].items():
        flat[f"model.{k}"] = v
    for group_name, group in weights["optimizer"].items():
        for k, v in group.items():
            if isinstance(v, mx.array):
                flat[f"optimizer.{group_name}.{k}"] = v

    mx.save_safetensors(base_path, flat, metadata={})

    sampler_state = train_sampler.state_dict()
    meta = {
        "step": global_step,
        "tokens_processed": tokens_processed,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "elapsed_time": elapsed_time,
        "rng_state": {
            "python": list(_capture_rng()["python"][1]),
            "python_version": _capture_rng()["python"][0],
        },
        "sampler": {
            "rng_state": _json_safe(sampler_state["rng_state"]),
            "indices": sampler_state["indices"],
            "position": sampler_state["position"],
            "dataset_size": sampler_state["dataset_size"],
            "batch_size": sampler_state["batch_size"],
            "drop_last": sampler_state["drop_last"],
        },
        "preset_name": config.preset_name,
    }
    save_meta_json(base_path + ".meta.json", meta)


def _json_safe(obj):
    """Make numpy RNG state JSON-serializable."""
    import numpy as _np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return {"__ndarray__": True, "dtype": str(obj.dtype), "data": obj.tolist()}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    return obj


def _json_restore(obj):
    import numpy as _np

    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return _np.asarray(obj["data"], dtype=_np.dtype(obj["dtype"]))
        return {k: _json_restore(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_restore(v) for v in obj]
    return obj


def load_training_checkpoint_mlx(base_path: str) -> dict:
    flat = load_safetensors(base_path)
    meta = load_meta_json(base_path + ".meta.json")

    model_flat: dict[str, mx.array] = {}
    opt_decay_flat: dict[str, mx.array] = {}
    opt_nodecay_flat: dict[str, mx.array] = {}
    for key, arr in flat.items():
        if key.startswith("model."):
            model_flat[key[len("model.") :]] = arr
        elif key.startswith("optimizer.decay."):
            opt_decay_flat[key[len("optimizer.decay.") :]] = arr
        elif key.startswith("optimizer.nodecay."):
            opt_nodecay_flat[key[len("optimizer.nodecay.") :]] = arr

    return {
        "model": tree_unflatten(list(model_flat.items())),
        "optimizer": {
            "decay": tree_unflatten(list(opt_decay_flat.items())),
            "nodecay": tree_unflatten(list(opt_nodecay_flat.items())),
        },
        "meta": meta,
    }


def evaluate(
    model: SpakieGPTMLX,
    val_dataset: PretrainDatasetMLX,
    val_sampler: ResumableBatchSamplerMLX,
    config: SpakieConfig,
) -> float:
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        count = 0
        it = iter(val_sampler)
        while count < config.pretrain_eval_batches:
            try:
                batch_indices = next(it)
            except StopIteration:
                break
            x_np, y_np = stack_batch(val_dataset, batch_indices)
            x = mx.array(x_np)
            y = mx.array(y_np)
            _, loss, _ = model(x, y)
            mx.eval(loss)
            total += float(loss.item())
            count += 1
    finally:
        if was_training:
            model.train()
    return total / max(count, 1)


def pretrain_mlx(
    model: SpakieGPTMLX,
    train_dataset: PretrainDatasetMLX,
    val_dataset: PretrainDatasetMLX,
    train_sampler: ResumableBatchSamplerMLX,
    config: SpakieConfig,
    runtime: MLXRuntimeSettings,
    resume_state: dict | None = None,
) -> float:
    if runtime.dtype != mx.float32:
        model.set_dtype(runtime.dtype)
    model.train()

    optimizer = DualAdamW(
        learning_rate=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
        betas=(0.9, 0.95),
    )

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    tokens_processed = 0
    target_tokens = config.pretrain_target_tokens
    elapsed_before_resume = 0.0
    last_val_loss = None
    stop_requested = False
    interrupted = False

    if resume_state:
        model.update(resume_state["model"])
        optimizer.load_state_trees(resume_state["optimizer"])
        meta = resume_state["meta"]
        best_val_loss = float(meta.get("best_val_loss", best_val_loss))
        global_step = int(meta.get("step", 0))
        tokens_processed = int(meta.get("tokens_processed", 0))
        elapsed_before_resume = float(meta.get("elapsed_time", 0.0))
        last_val_loss = meta.get("val_loss")

    val_sampler = ResumableBatchSamplerMLX(
        dataset_size=len(val_dataset), batch_size=config.pretrain_batch_size, drop_last=False, seed=1
    )

    accum_scale = 1.0 / config.pretrain_grad_accum_steps
    loss_and_grad = _build_loss_and_grad(model, accum_scale)

    start_time = time.time() - elapsed_before_resume
    pbar = tqdm(total=config.pretrain_max_steps, desc="Pretrain[mlx]", initial=global_step)

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        pbar.write("\nStop requested. Finishing the current optimizer step before saving resume state...")

    signal.signal(signal.SIGINT, handle_sigint)

    train_iter = iter(train_sampler)

    try:
        while global_step < config.pretrain_max_steps and (
            target_tokens <= 0 or tokens_processed < target_tokens
        ):
            accum_grads = None
            accum_loss = mx.array(0.0, dtype=mx.float32)

            for _ in range(config.pretrain_grad_accum_steps):
                try:
                    batch_indices = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_sampler)
                    batch_indices = next(train_iter)

                x_np, y_np = stack_batch(train_dataset, batch_indices)
                x = mx.array(x_np)
                y = mx.array(y_np)

                # Loss-fn has accum_scale baked in, so grads and loss are pre-scaled.
                loss, grads = loss_and_grad(model, x, y)
                accum_grads = _accum_grads(accum_grads, grads)
                accum_loss = accum_loss + loss.astype(mx.float32)
                # Materialize per microbatch: without this, the lazy graph grows
                # across the whole accumulation window and first-step tracing
                # overhead dominates — that's the MLX equivalent of forgetting
                # to call .backward() on each microbatch.
                mx.eval(accum_grads, accum_loss)
                tokens_processed += x.size

            clipped_grads, _ = clip_grads(accum_grads, config.pretrain_grad_clip)

            lr = get_lr(global_step, config)
            optimizer.set_lr(lr)
            optimizer.update(model, clipped_grads)

            mx.eval(model.parameters(), optimizer.decay.state, optimizer.nodecay.state, accum_loss)
            accum_loss_val = float(accum_loss.item())

            global_step += 1
            pbar.update(1)

            if stop_requested:
                interrupted = True
                interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")
                save_training_checkpoint_mlx(
                    interrupt_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    tokens_processed=tokens_processed,
                    best_val_loss=best_val_loss,
                    val_loss=last_val_loss,
                    elapsed_time=time.time() - start_time,
                    train_sampler=train_sampler,
                )
                pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
                break

            if global_step % config.pretrain_eval_interval == 0:
                val_loss = evaluate(model, val_dataset, val_sampler, config)
                last_val_loss = val_loss
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
                token_progress = ""
                if target_tokens > 0:
                    progress_pct = min(tokens_processed / target_tokens * 100.0, 100.0)
                    token_progress = (
                        f" | tokens {tokens_processed:,}/{target_tokens:,} ({progress_pct:.1f}%)"
                    )
                pbar.write(
                    f"step {global_step} | train_loss {accum_loss_val:.4f} | val_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_per_sec:.0f}{token_progress}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = os.path.join(config.checkpoint_dir, "pretrain_best.safetensors")
                    save_training_checkpoint_mlx(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        global_step=global_step,
                        tokens_processed=tokens_processed,
                        best_val_loss=best_val_loss,
                        val_loss=val_loss,
                        elapsed_time=elapsed,
                        train_sampler=train_sampler,
                    )
                    pbar.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")
                elif config.should_use_pretrain_early_stopping():
                    patience_counter += 1
                    if patience_counter >= config.pretrain_patience:
                        pbar.write(f"Early stopping at step {global_step}")
                        break

    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")
        save_training_checkpoint_mlx(
            interrupt_path,
            model=model,
            optimizer=optimizer,
            config=config,
            global_step=global_step,
            tokens_processed=tokens_processed,
            best_val_loss=best_val_loss,
            val_loss=last_val_loss,
            elapsed_time=time.time() - start_time,
            train_sampler=train_sampler,
        )
        pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        pbar.close()

    if interrupted:
        print(
            f"Pretraining stopped early at {tokens_processed:,} tokens. "
            f"Best val loss so far: {best_val_loss:.4f}"
        )
    else:
        print(
            f"Pretraining complete at {tokens_processed:,} tokens. Best val loss: {best_val_loss:.4f}"
        )
    return best_val_loss
