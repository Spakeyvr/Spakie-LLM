"""MLX pretraining loop: cosine LR, grad accumulation, resumable checkpoints."""

from __future__ import annotations

import atexit
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
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION, SpakieConfig, config_to_dict
from model.transformer_mlx import SpakieGPTMLX
from runtime.checkpoint_io import (
    checkpoint_processed_data_fingerprint,
    checkpoint_tokenizer_contract,
    load_mlx_checkpoint_config,
)
from runtime.mlx_backend import (
    MLXRuntimeSettings,
    clip_grads,
    load_safetensors,
    load_safetensors_checkpoint_meta,
    save_safetensors_checkpoint,
)
from training.dataset_mlx import (
    PretrainDatasetMLX,
    ResumableBatchSamplerMLX,
    stack_batch,
)
from training.mlx_profile import MLXProfile, now
from training.monitor import (
    TrainingStatusWriter,
    format_monitor_start_message,
    start_background_monitor,
    stop_background_monitor,
)
from training.muon_core import (
    MUON_OPTIMIZER_SCHEMA_VERSION,
    adamw_fallback_warning,
    should_adamw_fallback,
)
from training.optimizers_mlx import configure_mlx_optimizer
from training.prefetch_mlx import BatchPrefetcher


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
    def loss_fn(model, *batch):
        if len(batch) == 2:
            x, y = batch
            segment_ids = None
            position_ids = None
            loss_indices = None
            loss_targets = None
            loss_mask = None
            varlen_indices = None
            varlen_cu_seqlens = None
            valid_indices = None
            valid_mask = None
        elif len(batch) == 4:
            x, y = batch[:2]
            loss_indices = None
            loss_targets = None
            loss_mask = None
            varlen_indices = None
            varlen_cu_seqlens = None
            if batch[2].ndim == 1:
                valid_indices, valid_mask = batch[2], batch[3]
                segment_ids = None
                position_ids = None
            else:
                segment_ids, position_ids = batch[2], batch[3]
                valid_indices = None
                valid_mask = None
        elif len(batch) == 5:
            x, y, loss_indices, loss_targets, loss_mask = batch
            segment_ids = None
            position_ids = None
            varlen_indices = None
            varlen_cu_seqlens = None
            valid_indices = None
            valid_mask = None
        elif len(batch) == 6:
            x, y, segment_ids, position_ids, varlen_indices, varlen_cu_seqlens = batch
            loss_indices = None
            loss_targets = None
            loss_mask = None
            valid_indices = None
            valid_mask = None
        elif len(batch) == 7:
            varlen_indices = None
            varlen_cu_seqlens = None
            x, y = batch[:2]
            if batch[2].ndim == 1:
                valid_indices, valid_mask, loss_indices, loss_targets, loss_mask = batch[2:]
                segment_ids = None
                position_ids = None
            else:
                segment_ids, position_ids, loss_indices, loss_targets, loss_mask = batch[2:]
                valid_indices = None
                valid_mask = None
        elif len(batch) == 9:
            (
                x,
                y,
                segment_ids,
                position_ids,
                varlen_indices,
                varlen_cu_seqlens,
                loss_indices,
                loss_targets,
                loss_mask,
            ) = batch
            valid_indices = None
            valid_mask = None
        else:
            raise ValueError(f"unsupported MLX training batch with {len(batch)} arrays")
        _, loss, _ = model(
            x,
            y,
            return_cache=False,
            ignore_index=ignore_index,
            segment_ids=segment_ids,
            position_ids=position_ids,
            loss_indices=loss_indices,
            loss_targets=loss_targets,
            loss_mask=loss_mask,
            varlen_indices=varlen_indices,
            varlen_cu_seqlens=varlen_cu_seqlens,
            valid_indices=valid_indices,
            valid_mask=valid_mask,
        )
        return loss * accum_scale

    return nn.value_and_grad(model, loss_fn)


def _build_microbatch_step(
    model: SpakieGPTMLX,
    accum_scale: float,
    *,
    compile_step: bool,
    ignore_index: int | None = -100,
    shapeless: bool = False,
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
        def step(*batch):
            return value_and_grad(model, *batch)
        return step

    state = [model.state]
    if getattr(model.config, "dropout", 0.0) > 0.0:
        state.append(mx.random.state)

    @partial(mx.compile, inputs=state, outputs=state, shapeless=shapeless)
    def step(*batch):
        return value_and_grad(model, *batch)

    return step


def _build_vmap_accum_step(
    model: SpakieGPTMLX,
    *,
    compile_step: bool,
    ignore_index: int | None = -100,
    loss_scale: float | None = None,
):
    """Return a callable `step(xs, ys) -> (loss, grads)` for stacked microbatches.

    `xs` and `ys` have shape (lanes, batch, seq). With `loss_scale=None` the
    returned loss is the mean over the lanes in this call (matching a single
    full-G vmap of `grad_accum` microbatches). When the accumulation is split
    into several smaller groups, pass `loss_scale=1/grad_accum`: each group then
    returns `sum(lane_losses) / grad_accum`, so summing the group losses and
    group gradients across all groups reproduces the exact 1/grad_accum scaling
    of the full-G vmap (and of the sequential accumulation loop).
    """

    def loss_fn(model, xs, ys):
        def one_loss(x, y):
            _, loss, _ = model(
                x,
                y,
                return_cache=False,
                ignore_index=ignore_index,
                # This runs under mx.vmap; the fused CE Metal kernel has no vmap
                # rule, so force the vmap-safe MLX custom loss instead.
                allow_fused_ce=False,
            )
            return loss

        lane_losses = mx.vmap(one_loss)(xs, ys)
        if loss_scale is None:
            return lane_losses.mean()
        return lane_losses.sum() * loss_scale

    value_and_grad = nn.value_and_grad(model, loss_fn)

    if not compile_step:
        def step(xs, ys):
            return value_and_grad(model, xs, ys)
        return step

    state = [model.state]
    if getattr(model.config, "dropout", 0.0) > 0.0:
        state.append(mx.random.state)

    @partial(mx.compile, inputs=state, outputs=state)
    def step(xs, ys):
        return value_and_grad(model, xs, ys)

    return step


def _accum_grads(acc, new):
    if acc is None:
        return new
    return tree_map(operator.add, acc, new)


def _physical_ram_bytes() -> int:
    """Total physical RAM in bytes (unified memory on Apple Silicon)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        import subprocess

        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))


def _reset_peak_memory() -> None:
    for fn_name in ("reset_peak_memory", "clear_cache"):
        fn = getattr(mx, fn_name, None)
        if fn is not None:
            fn()


def choose_vmap_group_size(
    model: SpakieGPTMLX,
    config: SpakieConfig,
    *,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    budget_frac: float,
) -> int:
    """Pick the largest vmap group size whose forward/backward peak fits budget.

    vmap keeps every lane's activations resident for the backward, so peak memory
    grows ~linearly with the number of lanes vmapped together. A full-G vmap of
    the 300m preset needs ~107 GB, which panics the macOS kernel on a 128 GB
    machine. This runs one cheap single-lane probe at a small batch, measures the
    real per-(lane*batch) peak, and returns the largest group in ``1..grad_accum``
    whose estimated peak stays under ``budget_frac`` of physical RAM.

    A positive ``pretrain_vmap_group_size`` overrides the probe (still clamped to
    ``grad_accum``).
    """
    forced = int(getattr(config, "pretrain_vmap_group_size", 0) or 0)
    if forced > 0:
        return max(1, min(forced, grad_accum))

    budget = _physical_ram_bytes() * budget_frac
    probe_b = min(batch_size, 16)
    probe_step = _build_vmap_accum_step(
        model, compile_step=False, ignore_index=None, loss_scale=1.0
    )
    xs = mx.zeros((1, probe_b, seq_len), dtype=mx.int32)
    ys = mx.zeros((1, probe_b, seq_len), dtype=mx.int32)
    _reset_peak_memory()
    loss, grads = probe_step(xs, ys)
    mx.eval(loss, grads)
    probe_peak = mx.get_peak_memory()
    del loss, grads, probe_step, xs, ys
    _reset_peak_memory()

    # Peak is ~ group * batch * per_unit (base term is small and shared across
    # lanes, so treating it as per-unit slightly overestimates -> conservative).
    per_unit = probe_peak / max(probe_b, 1)
    best = 1
    for group in range(1, grad_accum + 1):
        estimate = group * batch_size * per_unit
        if estimate <= budget:
            best = group
    return best


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


class AsyncCheckpointWriter:
    """Serializes safetensors writes onto a worker thread.

    The producer (training loop) calls `submit(path, flat, meta)`
    after having eval'd the array dict on the main thread, so the arrays are
    materialized and safe to hand off. Only one write is in flight at a time;
    submitting a second one blocks until the first finishes. `join()` waits
    on the current write and is invoked before interpreter shutdown.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def submit(self, path: str, flat: dict[str, mx.array], meta: dict) -> None:
        self.join()

        def _work() -> None:
            try:
                save_safetensors_checkpoint(path, flat, meta)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc

        thread = threading.Thread(target=_work, daemon=False)
        self._thread = thread
        thread.start()

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
            "optimizer_schema_version": MUON_OPTIMIZER_SCHEMA_VERSION,
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_steps": config.muon_ns_steps,
            "ns_coefficients": list(config.muon_ns_coefficients),
            "eps": config.muon_eps,
            "adjust_lr_fn": config.muon_adjust_lr_fn,
            "qkv_split": config.muon_qkv_split,
        },
        "muon_verified": config.muon_verified,
        "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
        "config": config_to_dict(config),
        "tokenizer": checkpoint_tokenizer_contract(config),
        "processed_data_manifest_sha256": checkpoint_processed_data_fingerprint(config),
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
    """Write a self-contained .safetensors checkpoint with embedded metadata.

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
    if writer is not None and not sync:
        # Materialize on the main thread so the worker only does I/O.
        if flat:
            mx.eval(*flat.values())
        writer.submit(base_path, flat, meta)
        return

    if flat:
        mx.eval(*flat.values())
    save_safetensors_checkpoint(base_path, flat, meta)


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
    # Validate the exact current config contract before mapping large tensors.
    load_mlx_checkpoint_config(base_path)
    meta = load_safetensors_checkpoint_meta(base_path)
    if meta is None:
        raise ValueError(f"Missing metadata for MLX training checkpoint '{base_path}'")
    sampler_meta = meta.get("sampler", {})
    if isinstance(sampler_meta, dict) and "indices" in sampler_meta:
        raise ValueError(
            f"MLX checkpoint '{base_path}' stores sampler indices in obsolete metadata"
        )
    flat = load_safetensors(base_path)

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
        for batch_indices in val_sampler.iter_fixed(config.pretrain_eval_batches):
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
    eval_microbatch_loss: bool = True,
    eval_loss_final_microbatch: bool = False,
    defer_final_microbatch_eval: bool = False,
    use_vmap_accum_step: bool = False,
    allow_adamw_fallback: bool = False,
) -> float:
    if resume_state:
        model.load_weights(tree_flatten(resume_state["model"]), strict=True)
    # Loading replaces arrays with checkpoint dtypes. Cast after loading even
    # for explicit FP32 so the requested runtime precision is authoritative.
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
    vmap_accum_step = None
    vmap_group_size = 0
    vmap_n_groups = 0
    vmap_sync_warmup = 0
    if use_vmap_accum_step:
        if config.pretrain_grad_accum_steps < 2:
            raise ValueError("use_vmap_accum_step requires pretrain_grad_accum_steps >= 2")
        grad_accum = config.pretrain_grad_accum_steps
        vmap_group_size = choose_vmap_group_size(
            model,
            config,
            batch_size=config.pretrain_batch_size,
            seq_len=config.max_seq_len,
            grad_accum=grad_accum,
            budget_frac=float(getattr(config, "pretrain_vmap_mem_budget_frac", 0.70)),
        )
        vmap_n_groups = math.ceil(grad_accum / vmap_group_size)
        vmap_sync_warmup = int(
            getattr(config, "pretrain_vmap_sync_warmup_steps", 0) or 0
        )
        print(
            f"vmap accumulation: grad_accum={grad_accum}, group_size={vmap_group_size}, "
            f"groups/step={vmap_n_groups} (peak ~ group_size lanes resident; "
            f"a full-G vmap would hold all {grad_accum} lanes and can panic the "
            f"macOS kernel when it exceeds physical RAM)"
        )
        # loss_scale = 1/grad_accum so summing group losses/grads across all
        # groups reproduces the exact 1/grad_accum scaling of a full-G vmap.
        vmap_accum_step = _build_vmap_accum_step(
            model,
            compile_step=use_compile,
            ignore_index=None,
            loss_scale=1.0 / grad_accum,
        )

    ckpt_writer = AsyncCheckpointWriter()
    profiler = MLXProfile(enabled=profile)
    window_steps = 0

    start_time = time.time() - elapsed_before_resume
    status_writer = TrainingStatusWriter(
        config.checkpoint_dir,
        stage="pretrain",
        backend="mlx",
        preset=config.preset_name,
        total_steps=config.pretrain_max_steps,
        target_tokens=target_tokens,
        elapsed_offset=elapsed_before_resume,
    )
    status_writer.update(
        force=True,
        status="running",
        step=global_step,
        tokens_processed=tokens_processed,
        best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        val_loss=last_val_loss,
    )
    monitor_info = start_background_monitor(status_writer.path, config.checkpoint_dir)
    print(format_monitor_start_message(monitor_info))
    atexit.register(stop_background_monitor, monitor_info)
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
    committed_sampler = ResumableBatchSamplerMLX.from_state_dict(
        train_sampler.state_dict(copy_indices=False)
    )
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
            consumed_microbatches = 0
            step_tokens = 0

            if vmap_accum_step is not None:
                xs = []
                ys = []
                for _ in range(config.pretrain_grad_accum_steps):
                    if profiler.enabled:
                        batch_start = now()
                        x_np, y_np = next_batch()
                        profiler.add("batch_fetch", now() - batch_start)
                    else:
                        x_np, y_np = next_batch()
                    x, y = _arrays_to_mx(x_np, y_np, profiler)
                    xs.append(x)
                    ys.append(y)
                    step_tokens += x.size
                    consumed_microbatches += 1

                step_start = now() if profiler.enabled else None
                grad_accum = config.pretrain_grad_accum_steps
                for gi in range(vmap_n_groups):
                    lo = gi * vmap_group_size
                    hi = min(lo + vmap_group_size, grad_accum)
                    group_loss, group_grads = vmap_accum_step(
                        mx.stack(xs[lo:hi]), mx.stack(ys[lo:hi])
                    )
                    accum_grads = (
                        group_grads
                        if accum_grads is None
                        else _accum_grads(accum_grads, group_grads)
                    )
                    accum_loss = accum_loss + group_loss.astype(mx.float32)
                    # Sync between groups so each group's forward activations are
                    # freed before the next group's graph is built. Without this
                    # the groups overlap and peak memory returns to the full-G
                    # (kernel-panicking) footprint. The final group may async_eval
                    # to overlap with the optimizer, except during sync warmup.
                    is_last = gi == vmap_n_groups - 1
                    if is_last and global_step >= vmap_sync_warmup:
                        mx.async_eval(accum_grads, accum_loss)
                    else:
                        mx.eval(accum_grads, accum_loss)
                if profiler.enabled:
                    profiler.add("forward_backward", now() - step_start)
            else:
                for micro_idx in range(config.pretrain_grad_accum_steps):
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
                        if eval_microbatch_loss:
                            mx.eval(loss, grads)
                        else:
                            mx.eval(grads)
                        profiler.add("forward_backward", now() - step_start)
                    else:
                        loss, grads = microbatch_step(x, y)
                    accum_grads = _accum_grads(accum_grads, grads)
                    accum_loss = accum_loss + loss.astype(mx.float32)
                    # Materialize after each microbatch so MLX frees the per-microbatch
                    # activation/intermediate graph. async_eval lets Python queue the
                    # next batch while Metal drains the current graph.
                    should_eval_microbatch = not (
                        defer_final_microbatch_eval
                        and micro_idx == config.pretrain_grad_accum_steps - 1
                    )
                    if should_eval_microbatch:
                        include_loss = eval_microbatch_loss or (
                            eval_loss_final_microbatch
                            and micro_idx == config.pretrain_grad_accum_steps - 1
                        )
                        eval_target = (
                            (accum_grads, accum_loss)
                            if include_loss
                            else accum_grads
                        )
                        mx.async_eval(eval_target)
                    step_tokens += x.size
                    consumed_microbatches += 1

            if profiler.enabled:
                opt_start = now()
                clipped_grads, _ = (
                    (accum_grads, None)
                    if config.pretrain_grad_clip <= 0
                    else clip_grads(accum_grads, config.pretrain_grad_clip)
                )
            else:
                clipped_grads, _ = (
                    (accum_grads, None)
                    if config.pretrain_grad_clip <= 0
                    else clip_grads(accum_grads, config.pretrain_grad_clip)
                )

            lr = get_lr(global_step, config)
            optimizer.set_lr(lr)
            try:
                optimizer.update(model, clipped_grads)
            except Exception as exc:
                if should_adamw_fallback(
                    exc, optimizer, config, stage="pretrain", allow=allow_adamw_fallback
                ):
                    print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
                    config.pretrain_optimizer = "adamw"
                    print(adamw_fallback_warning("Pretraining"))
                    optimizer = configure_mlx_optimizer(
                        model,
                        config,
                        kind="adamw",
                        learning_rate=config.pretrain_lr,
                        weight_decay=config.pretrain_weight_decay,
                    )
                    optimizer.set_lr(lr)
                    optimizer.update(model, clipped_grads)
                else:
                    raise

            mx.eval(*optimizer.evaluation_state(), accum_loss)
            accum_loss_val = None
            if profiler.enabled:
                profiler.add("opt_step", now() - opt_start)

            global_step += 1
            tokens_processed += step_tokens
            committed_sampler.advance_batches(consumed_microbatches)
            window_steps += 1
            pbar.update(1)
            if status_writer.due():
                accum_loss_val = float(accum_loss.item())
                elapsed = time.time() - start_time
                status_writer.update(
                    status="running",
                    step=global_step,
                    total_steps=config.pretrain_max_steps,
                    train_loss=accum_loss_val,
                    val_loss=last_val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                    lr=lr,
                    tokens_processed=tokens_processed,
                    target_tokens=target_tokens,
                    tok_per_sec=tokens_processed / elapsed if elapsed > 0 else 0.0,
                )

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
                        train_sampler=committed_sampler,
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
                        train_sampler=committed_sampler,
                        writer=ckpt_writer,
                        sync=True,
                    )
                status_writer.finish(
                    "interrupted",
                    message=f"Saved checkpoint to {interrupt_path}",
                    step=global_step,
                    tokens_processed=tokens_processed,
                    val_loss=last_val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                )
                pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
                break

            checkpoint_interval = int(getattr(config, "pretrain_checkpoint_interval", 0) or 0)
            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                rolling_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")
                if profiler.enabled:
                    checkpoint_start = now()
                    save_training_checkpoint_mlx(
                        rolling_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        global_step=global_step,
                        tokens_processed=tokens_processed,
                        best_val_loss=best_val_loss,
                        val_loss=last_val_loss,
                        elapsed_time=time.time() - start_time,
                        train_sampler=committed_sampler,
                        writer=ckpt_writer,
                        sync=True,
                    )
                    profiler.add("checkpoint", now() - checkpoint_start)
                else:
                    save_training_checkpoint_mlx(
                        rolling_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        global_step=global_step,
                        tokens_processed=tokens_processed,
                        best_val_loss=best_val_loss,
                        val_loss=last_val_loss,
                        elapsed_time=time.time() - start_time,
                        train_sampler=committed_sampler,
                        writer=ckpt_writer,
                        sync=True,
                    )
                status_writer.update(
                    force=True,
                    status="running",
                    step=global_step,
                    tokens_processed=tokens_processed,
                    val_loss=last_val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                    last_checkpoint=rolling_path,
                )
                pbar.write(f"  -> saved rolling checkpoint at step {global_step}")

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
                active_gb = mx.get_active_memory() / 1024**3
                cache_gb = mx.get_cache_memory() / 1024**3
                peak_gb = mx.get_peak_memory() / 1024**3
                pbar.write(
                    f"step {global_step} | train_loss {accum_loss_val:.4f} | val_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_per_sec:.0f}{token_progress} | "
                    f"mem active {active_gb:.1f}GB cache {cache_gb:.1f}GB peak {peak_gb:.1f}GB"
                )
                status_writer.update(
                    force=True,
                    status="running",
                    step=global_step,
                    train_loss=accum_loss_val,
                    val_loss=val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                    lr=lr,
                    tokens_processed=tokens_processed,
                    target_tokens=target_tokens,
                    tok_per_sec=tok_per_sec,
                    mlx_active_gb=active_gb,
                    mlx_cache_gb=cache_gb,
                    mlx_peak_gb=peak_gb,
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
                            train_sampler=committed_sampler,
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
                            train_sampler=committed_sampler,
                            writer=ckpt_writer,
                        )
                    status_writer.update(
                        force=True,
                        best_val_loss=best_val_loss,
                        best_checkpoint=ckpt_path,
                    )
                    pbar.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")
                elif config.should_use_pretrain_early_stopping():
                    patience_counter += 1
                    if patience_counter >= config.pretrain_patience:
                        pbar.write(f"Early stopping at step {global_step}")
                        status_writer.finish(
                            "stopped",
                            message=f"Early stopping at step {global_step}",
                            step=global_step,
                            train_loss=accum_loss_val,
                            val_loss=last_val_loss,
                            best_val_loss=best_val_loss,
                            tokens_processed=tokens_processed,
                        )
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
                train_sampler=committed_sampler,
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
                train_sampler=committed_sampler,
                writer=ckpt_writer,
                sync=True,
            )
        status_writer.finish(
            "interrupted",
            message=f"Saved checkpoint to {interrupt_path}",
            step=global_step,
            tokens_processed=tokens_processed,
            val_loss=last_val_loss,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
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
        final_path = os.path.join(config.checkpoint_dir, "pretrain_final.safetensors")
        save_training_checkpoint_mlx(
            final_path,
            model=model,
            optimizer=optimizer,
            config=config,
            global_step=global_step,
            tokens_processed=tokens_processed,
            best_val_loss=best_val_loss,
            val_loss=last_val_loss,
            elapsed_time=time.time() - start_time,
            train_sampler=committed_sampler,
            writer=ckpt_writer,
            sync=True,
        )
        if status_writer.payload.get("status") != "stopped":
            status_writer.finish(
                "complete",
                message="Pretraining complete",
                step=global_step,
                tokens_processed=tokens_processed,
                val_loss=last_val_loss,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            )
        print(
            f"Pretraining complete at {tokens_processed:,} tokens. Best val loss: {best_val_loss:.4f}"
        )
        print(f"Final checkpoint: {final_path}")
    if stop_background_monitor(monitor_info):
        print("Monitor stopped.")
    return best_val_loss
