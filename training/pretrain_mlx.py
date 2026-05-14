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
from training.mlx_profile import MLXProfile, now
from training.optimizers_mlx import configure_mlx_optimizer
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
    """Learning rate at `step`.

    Cosine: linear warmup -> cosine decay to 10% of peak.
    Trapezoid: linear warmup -> constant peak -> linear decay to 10% of peak
        across the final `pretrain_trapezoid_decay_frac` of training. On a
        controlled synthetic next-token learning task, trapezoid drove final
        loss measurably lower than cosine for the same compute budget.
    """
    min_lr = config.pretrain_lr * 0.1
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / max(config.pretrain_warmup_steps, 1)
    if step >= config.pretrain_max_steps:
        return min_lr
    schedule = getattr(config, "pretrain_lr_schedule", "cosine")
    if schedule == "trapezoid":
        decay_frac = max(0.0, min(1.0, getattr(config, "pretrain_trapezoid_decay_frac", 0.2)))
        decay_start = int(config.pretrain_max_steps * (1.0 - decay_frac))
        if step < decay_start:
            return config.pretrain_lr
        progress = (step - decay_start) / max(config.pretrain_max_steps - decay_start, 1)
        return config.pretrain_lr - progress * (config.pretrain_lr - min_lr)
    progress = (step - config.pretrain_warmup_steps) / (
        config.pretrain_max_steps - config.pretrain_warmup_steps
    )
    return min_lr + 0.5 * (config.pretrain_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _build_loss_and_grad(
    model: SpakieGPTMLX, accum_scale: float, *, ignore_index: int | None
):
    """Build value_and_grad with the grad-accum scale baked into the loss.

    Pre-scaling here means grads come out already divided by accum_steps — no
    Python-side tree_map needed between microbatches, which keeps the lazy
    graph small. `ignore_index=None` skips the SFT-style mask path entirely —
    use this for pretraining where targets never contain -100; it avoids
    materializing two extra (B*T) tensors per microbatch.
    """
    def loss_fn(model, x, y):
        _, loss, _ = model(x, y, return_cache=False, ignore_index=ignore_index)
        return loss * accum_scale

    return nn.value_and_grad(model, loss_fn)


def _build_microbatch_step(
    model: SpakieGPTMLX,
    accum_scale: float,
    *,
    compile_step: bool,
    ignore_index: int | None = -100,
):
    """Return a callable `step(x, y) -> (loss, grads)`.

    With `compile_step=True`, the forward+backward is wrapped in `mx.compile`
    with model state + global RNG state captured so dropout stays stochastic
    and parameter updates are observed across calls. Keeping `accum_scale` as
    a Python float means it becomes a compile-time constant — no recompile
    per step.
    """
    value_and_grad = _build_loss_and_grad(model, accum_scale, ignore_index=ignore_index)

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


def _arrays_to_mx(
    x_np: np.ndarray,
    y_np: np.ndarray,
    profiler: MLXProfile,
) -> tuple[mx.array, mx.array]:
    if not profiler.enabled:
        return mx.array(x_np), mx.array(y_np)
    start = now()
    x = mx.array(x_np)
    y = mx.array(y_np)
    mx.eval(x, y)
    profiler.add("numpy_to_mlx", now() - start)
    return x, y


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
    optimizer,
    config: SpakieConfig,
    global_step: int,
    tokens_processed: int,
    best_val_loss: float,
    val_loss: float | None,
    elapsed_time: float,
    train_sampler: ResumableBatchSamplerMLX,
) -> tuple[dict[str, mx.array], dict]:
    model_flat = dict(tree_flatten(model.parameters()))
    opt_state = optimizer.state_trees()

    flat: dict[str, mx.array] = {}
    for k, v in model_flat.items():
        flat[f"model.{k}"] = v
    for section_name, section_tree in opt_state.items():
        for k, v in tree_flatten(section_tree):
            if isinstance(v, mx.array):
                flat[f"optimizer.{section_name}.{k}"] = v

    sampler_state = train_sampler.state_dict(copy_indices=False)
    # Keep the large permutation out of JSON; dumping millions of ints as text
    # makes Ctrl+C checkpointing look like it is hanging after the final step.
    #
    # MLX array dimensions are int-limited, so multi-billion-sequence corpora
    # cannot store the sampler permutation as one tensor. In that case the
    # checkpoint remains resumable, but the sampler restarts with a new
    # deterministic shuffle instead of an exact in-epoch position.
    sampler_indices = np.asarray(sampler_state["indices"], dtype=np.int64)
    sampler_indices_format = "safetensors:sampler.indices"
    sampler_resume_exact = True
    max_mlx_dim = np.iinfo(np.int32).max
    if sampler_indices.size <= max_mlx_dim:
        flat["sampler.indices"] = mx.array(sampler_indices)
    else:
        sampler_indices_format = "omitted:too_large_for_mlx_array"
        sampler_resume_exact = False
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
            "position": sampler_state["position"],
            "dataset_size": sampler_state["dataset_size"],
            "batch_size": sampler_state["batch_size"],
            "drop_last": sampler_state["drop_last"],
            "indices_format": sampler_indices_format,
            "resume_exact": sampler_resume_exact,
        },
        "preset_name": config.preset_name,
        "optimizer_kind": getattr(optimizer, "optimizer_kind", config.pretrain_optimizer),
        "optimizer_warning": "fallback_not_recommended"
        if getattr(optimizer, "optimizer_kind", config.pretrain_optimizer) == "adamw"
        else "",
        "muon_hyperparameters": {
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_steps": config.muon_ns_steps,
            "ns_coefficients": list(config.muon_ns_coefficients),
            "eps": config.muon_eps,
            "adjust_lr_fn": config.muon_adjust_lr_fn,
            "qkv_split": config.muon_qkv_split,
        },
        "muon_verified": config.muon_verified,
    }
    return flat, meta


def save_training_checkpoint_mlx(
    base_path: str,
    *,
    model: SpakieGPTMLX,
    optimizer,
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
    optimizer_sections: dict[str, dict[str, mx.array]] = {}
    sampler_indices = None
    for key, arr in flat.items():
        if key.startswith("model."):
            model_flat[key[len("model.") :]] = arr
        elif key.startswith("optimizer."):
            _, section, rest = key.split(".", 2)
            optimizer_sections.setdefault(section, {})[rest] = arr
        elif key == "sampler.indices":
            sampler_indices = np.asarray(arr, dtype=np.int64)

    # New checkpoints store sampler indices in safetensors. Older checkpoints
    # still have them in meta["sampler"]["indices"], which remains supported.
    if sampler_indices is not None:
        meta.setdefault("sampler", {})["indices"] = sampler_indices

    return {
        "model": tree_unflatten(list(model_flat.items())),
        "optimizer": {
            section: tree_unflatten(list(values.items()))
            for section, values in optimizer_sections.items()
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
            _, loss, _ = model(x, y, return_cache=False, ignore_index=None)
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
    profile: bool = False,
) -> float:
    if runtime.dtype != mx.float32:
        model.set_dtype(runtime.dtype)
    model.train()

    optimizer = configure_mlx_optimizer(
        model,
        config,
        kind=config.pretrain_optimizer,
        learning_rate=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
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
        if "optimizer" in resume_state:
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
    # Pretraining never emits ignore_index tokens; skip the mask path to avoid
    # two extra (B*T) tensors per microbatch and a softer backward graph.
    microbatch_step = _build_microbatch_step(
        model, accum_scale, compile_step=use_compile, ignore_index=None
    )

    ckpt_writer = AsyncCheckpointWriter()
    profiler = MLXProfile(enabled=profile)
    window_steps = 0

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
            if profiler.enabled:
                checkpoint_start = now()
                ckpt_writer.join()
                profiler.add("checkpoint", now() - checkpoint_start)
            else:
                ckpt_writer.join()
            accum_grads = None
            accum_loss = mx.array(0.0, dtype=mx.float32)

            for _ in range(config.pretrain_grad_accum_steps):
                if profiler.enabled:
                    batch_start = now()
                    x_np, y_np = next_batch()
                    profiler.add("batch_fetch", now() - batch_start)
                else:
                    x_np, y_np = next_batch()
                x, y = _arrays_to_mx(x_np, y_np, profiler)

                # Loss-fn has accum_scale baked in, so grads and loss are pre-scaled.
                if profiler.enabled:
                    step_start = now()
                    loss, grads = microbatch_step(x, y)
                    # Sync only when profiling, to attribute time to forward_backward.
                    # In normal runs we let the graph extend across all microbatches
                    # so the GPU can pipeline fwd/bwd of microbatch N+1 behind the
                    # tail of microbatch N — the materialization point becomes the
                    # single mx.eval after the optimizer update.
                    mx.eval(loss, grads)
                    profiler.add("forward_backward", now() - step_start)
                else:
                    loss, grads = microbatch_step(x, y)
                accum_grads = _accum_grads(accum_grads, grads)
                accum_loss = accum_loss + loss.astype(mx.float32)
                tokens_processed += x.size

            if profiler.enabled:
                opt_start = now()
                clipped_grads, _ = clip_grads(accum_grads, config.pretrain_grad_clip)
            else:
                clipped_grads, _ = clip_grads(accum_grads, config.pretrain_grad_clip)

            lr = get_lr(global_step, config)
            optimizer.set_lr(lr)
            optimizer.update(model, clipped_grads)

            mx.eval(model.parameters(), optimizer.state_trees())
            accum_loss_val = None
            if profiler.enabled:
                profiler.add("opt_step", now() - opt_start)

            global_step += 1
            window_steps += 1
            pbar.update(1)

            if stop_requested:
                interrupted = True
                interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")
                if profiler.enabled:
                    checkpoint_start = now()
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
                    profiler.add("checkpoint", now() - checkpoint_start)
                else:
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
                if profiler.enabled:
                    eval_start = now()
                    val_loss = evaluate(model, val_dataset, val_sampler, config)
                    profiler.add("eval", now() - eval_start)
                else:
                    val_loss = evaluate(model, val_dataset, val_sampler, config)
                mx.eval(accum_loss)
                accum_loss_val = float(accum_loss.item())
                last_val_loss = val_loss
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
                token_progress = ""
                should_stop = False
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
                    if profiler.enabled:
                        checkpoint_start = now()
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
                        profiler.add("checkpoint", now() - checkpoint_start)
                    else:
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
                        should_stop = True
                if profiler.enabled:
                    pbar.write(
                        profiler.format_report(
                            window_label=f"last {window_steps} optimizer steps"
                        )
                    )
                    profiler.reset()
                    window_steps = 0
                if should_stop:
                    break

    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")
        if profiler.enabled:
            checkpoint_start = now()
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
            profiler.add("checkpoint", now() - checkpoint_start)
        else:
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
        if profiler.enabled:
            checkpoint_start = now()
            ckpt_writer.join()
            profiler.add("checkpoint", now() - checkpoint_start)
        else:
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
