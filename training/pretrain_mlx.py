"""MLX pretraining loop: cosine LR, grad accumulation, resumable checkpoints."""

from __future__ import annotations

import math
import operator
import os
import random
import signal
import sys
import threading
import time
from functools import partial

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
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
)
from training.dataset_mlx import (
    PretrainDatasetMLX,
    ResumableBatchSamplerMLX,
    stack_batch,
)
from training.prefetch_mlx import BatchPrefetcher


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


def _build_microbatch_step(model: SpakieGPTMLX, accum_scale: float, *, compile_step: bool):
    """Return a callable `step(x, y) -> (loss, grads)`.

    With `compile_step=True`, the forward+backward is wrapped in `mx.compile`
    with model state + global RNG state captured so dropout stays stochastic
    and parameter updates are observed across calls. Keeping `accum_scale` as
    a Python float means it becomes a compile-time constant — no recompile
    per step.
    """
    value_and_grad = _build_loss_and_grad(model, accum_scale)

    if not compile_step:
        def step(x, y):
            return value_and_grad(model, x, y)
        return step

    state = [model.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        return value_and_grad(model, x, y)

    return step


def _accum_grads(acc, new):
    if acc is None:
        return new
    return tree_map(operator.add, acc, new)


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


class AsyncCheckpointWriter:
    """Serializes safetensors writes onto a worker thread.

    The producer (training loop) calls `submit(path, flat, meta, meta_path)`
    after having eval'd the array dict on the main thread, so the arrays are
    materialized and safe to hand off. Only one write is in flight at a time;
    submitting a second one blocks until the first finishes. `join()` waits
    on the current write and is invoked before interpreter shutdown.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(self, path: str, flat: dict[str, mx.array], meta: dict, meta_path: str) -> None:
        self.join()

        def _work() -> None:
            try:
                mx.save_safetensors(path, flat, metadata={})
                save_meta_json(meta_path, meta)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc

        thread = threading.Thread(target=_work, daemon=False)
        self._thread = thread
        thread.start()

    def write_sync(self, path: str, flat: dict[str, mx.array], meta: dict, meta_path: str) -> None:
        self.join()
        if flat:
            mx.eval(*flat.values())
        mx.save_safetensors(path, flat, metadata={})
        save_meta_json(meta_path, meta)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise err


def _build_checkpoint_payload(
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
) -> tuple[dict[str, mx.array], dict]:
    model_flat = dict(tree_flatten(model.parameters()))
    opt_decay_flat = dict(tree_flatten(optimizer.decay.state))
    opt_nodecay_flat = dict(tree_flatten(optimizer.nodecay.state))

    flat: dict[str, mx.array] = {}
    for k, v in model_flat.items():
        flat[f"model.{k}"] = v
    for k, v in opt_decay_flat.items():
        if isinstance(v, mx.array):
            flat[f"optimizer.decay.{k}"] = v
    for k, v in opt_nodecay_flat.items():
        if isinstance(v, mx.array):
            flat[f"optimizer.nodecay.{k}"] = v

    sampler_state = train_sampler.state_dict()
    rng_snapshot = _capture_rng()
    meta = {
        "step": global_step,
        "tokens_processed": tokens_processed,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "elapsed_time": elapsed_time,
        "rng_state": {
            "python": list(rng_snapshot["python"][1]),
            "python_version": rng_snapshot["python"][0],
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
    return flat, meta


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
    writer: AsyncCheckpointWriter | None = None,
    sync: bool = False,
) -> None:
    """Write a .safetensors weights file plus a sibling .meta.json.

    When `writer` is provided and `sync=False`, the actual I/O happens on a
    worker thread; the caller is responsible for keeping the writer alive and
    joining it before exit. When `sync=True` (or no writer given) the save is
    synchronous — used for interrupt/final saves where we want the file
    flushed before returning.
    """
    flat, meta = _build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        global_step=global_step,
        tokens_processed=tokens_processed,
        best_val_loss=best_val_loss,
        val_loss=val_loss,
        elapsed_time=elapsed_time,
        train_sampler=train_sampler,
    )
    meta_path = base_path + ".meta.json"

    if writer is not None and not sync:
        # Materialize on the main thread so the worker only does I/O.
        if flat:
            mx.eval(*flat.values())
        writer.submit(base_path, flat, meta, meta_path)
        return

    if flat:
        mx.eval(*flat.values())
    mx.save_safetensors(base_path, flat, metadata={})
    save_meta_json(meta_path, meta)


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
    *,
    use_compile: bool = True,
    use_prefetch: bool = True,
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
    microbatch_step = _build_microbatch_step(model, accum_scale, compile_step=use_compile)

    ckpt_writer = AsyncCheckpointWriter()

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

    prefetcher: BatchPrefetcher | None = None
    train_iter = None
    if use_prefetch:
        prefetcher = BatchPrefetcher(train_dataset, train_sampler)
    else:
        train_iter = iter(train_sampler)

    def next_batch() -> tuple[np.ndarray, np.ndarray]:
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
        while global_step < config.pretrain_max_steps and (
            target_tokens <= 0 or tokens_processed < target_tokens
        ):
            # Ensure any in-flight async checkpoint write finishes before we
            # issue new Metal ops — mx.save_safetensors on the writer thread
            # races with the main-thread Metal scheduler otherwise (segfault).
            ckpt_writer.join()
            accum_grads = None
            accum_loss = mx.array(0.0, dtype=mx.float32)

            for _ in range(config.pretrain_grad_accum_steps):
                x_np, y_np = next_batch()
                x = mx.array(x_np)
                y = mx.array(y_np)

                # Loss-fn has accum_scale baked in, so grads and loss are pre-scaled.
                loss, grads = microbatch_step(x, y)
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
                    writer=ckpt_writer,
                    sync=True,
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
                        writer=ckpt_writer,
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
            writer=ckpt_writer,
            sync=True,
        )
        pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        pbar.close()
        if prefetcher is not None:
            prefetcher.close()
        ckpt_writer.join()

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
