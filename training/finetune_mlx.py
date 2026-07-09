"""MLX SFT fine-tuning loop."""

from __future__ import annotations

import atexit
import math
import os
import sys
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION, SpakieConfig, config_to_dict
from model.transformer_mlx import SpakieGPTMLX
from runtime.mlx_backend import (
    MLXRuntimeSettings,
    clip_grads,
    save_safetensors_checkpoint,
)
from training.dataset_mlx import (
    append_packed_varlen_attention_metadata,
    append_supervised_loss_indices,
    append_valid_token_indices,
    HomogeneousStepSortedBatchSamplerMLX,
    LengthBucketBatchSamplerMLX,
    pack_sft_batch,
    sequence_lengths,
    SortedLengthBatchSamplerMLX,
    stack_batch,
    trim_right_padding_bucket,
)
from training.mlx_profile import MLXProfile, now
from training.monitor import (
    TrainingStatusWriter,
    format_monitor_start_message,
    start_background_monitor,
    stop_background_monitor,
)
from training.muon_core import adamw_fallback_warning, should_adamw_fallback
from training.optimizers_mlx import configure_mlx_optimizer
from training.pretrain_mlx import _accum_grads, _build_microbatch_step
from training.prefetch_mlx import BatchPrefetcher


def _save_sft_checkpoint(
    base_path: str,
    model: SpakieGPTMLX,
    meta: dict,
    *,
    config: SpakieConfig,
    optimizer=None,
) -> None:
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    flat = {f"model.{k}": v for k, v in tree_flatten(model.parameters())}
    if optimizer is not None:
        opt_state = optimizer.state_trees()
        for section_name, section_tree in opt_state.items():
            for key, value in tree_flatten(section_tree):
                if isinstance(value, mx.array):
                    flat[f"optimizer.{section_name}.{key}"] = value
    if flat:
        mx.eval(*flat.values())
    meta = dict(meta)
    meta["config_schema_version"] = CHECKPOINT_CONFIG_SCHEMA_VERSION
    meta["config"] = config_to_dict(config)
    save_safetensors_checkpoint(base_path, flat, meta)


def _prepare_sft_batch_np(
    batch_np: tuple,
    *,
    config: SpakieConfig,
    sft_pack: bool,
    sft_gather_loss: bool,
    sft_bucket_multiple: int,
) -> tuple:
    x_np, y_np = batch_np[:2]
    if sft_pack:
        x_np, y_np = trim_right_padding_bucket(
            x_np, y_np, bucket_multiple=sft_bucket_multiple
        )
        batch_np = pack_sft_batch(x_np, y_np, max_seq_len=x_np.shape[1])
        if getattr(config, "attention_backend", "sdpa") == "mfa-varlen":
            batch_np = append_packed_varlen_attention_metadata(batch_np)
        if sft_gather_loss:
            batch_np = append_supervised_loss_indices(
                batch_np, bucket_multiple=sft_bucket_multiple
            )
    else:
        batch_np = trim_right_padding_bucket(
            x_np, y_np, bucket_multiple=sft_bucket_multiple
        )
        if config.compact_valid_mlp or config.compact_valid_projections:
            batch_np = append_valid_token_indices(
                batch_np, bucket_multiple=sft_bucket_multiple
            )
        if sft_gather_loss:
            batch_np = append_supervised_loss_indices(
                batch_np, bucket_multiple=sft_bucket_multiple
            )
    return batch_np


def _forward_sft_eval_loss(model: SpakieGPTMLX, batch_np: tuple, *, use_varlen: bool) -> mx.array:
    batch_mx = tuple(mx.array(array) for array in batch_np)
    if use_varlen and len(batch_mx) >= 6:
        x, y, segment_ids, position_ids, varlen_indices, varlen_cu_seqlens = batch_mx[:6]
        _, loss, _ = model(
            x,
            y,
            return_cache=False,
            ignore_index=-100,
            segment_ids=segment_ids,
            position_ids=position_ids,
            varlen_indices=varlen_indices,
            varlen_cu_seqlens=varlen_cu_seqlens,
        )
        return loss
    if len(batch_mx) == 7 and batch_mx[2].ndim == 1:
        x, y, valid_indices, valid_mask, loss_indices, loss_targets, loss_mask = batch_mx
        _, loss, _ = model(
            x,
            y,
            return_cache=False,
            ignore_index=-100,
            valid_indices=valid_indices,
            valid_mask=valid_mask,
            loss_indices=loss_indices,
            loss_targets=loss_targets,
            loss_mask=loss_mask,
        )
        return loss
    if len(batch_mx) == 7:
        x, y, segment_ids, position_ids, loss_indices, loss_targets, loss_mask = batch_mx
        _, loss, _ = model(
            x,
            y,
            return_cache=False,
            ignore_index=-100,
            segment_ids=segment_ids,
            position_ids=position_ids,
            loss_indices=loss_indices,
            loss_targets=loss_targets,
            loss_mask=loss_mask,
        )
        return loss
    if len(batch_mx) == 5:
        x, y, loss_indices, loss_targets, loss_mask = batch_mx
        _, loss, _ = model(
            x,
            y,
            return_cache=False,
            ignore_index=-100,
            loss_indices=loss_indices,
            loss_targets=loss_targets,
            loss_mask=loss_mask,
        )
        return loss
    x, y = batch_mx[:2]
    _, loss, _ = model(x, y, return_cache=False, ignore_index=-100)
    return loss


def _scale_partial_accum_grads(accum_grads, config: SpakieConfig, micro_in_step: int):
    if micro_in_step <= 0:
        raise ValueError("micro_in_step must be positive")
    if micro_in_step == config.sft_grad_accum_steps:
        return accum_grads
    scale = config.sft_grad_accum_steps / micro_in_step
    return tree_map(lambda grad: grad * scale, accum_grads)


def _evaluate_sft_loss(
    model: SpakieGPTMLX,
    val_dataset,
    batch_size: int,
    *,
    config: SpakieConfig,
    sft_pack: bool,
    sft_gather_loss: bool,
    sft_bucket_multiple: int,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_varlen = sft_pack and config.attention_backend == "mfa-varlen"

    try:
        for start in range(0, len(val_dataset), batch_size):
            stop = min(start + batch_size, len(val_dataset))
            batch_np = stack_batch(val_dataset, range(start, stop))
            valid_tokens = int((batch_np[1] != -100).sum())
            if valid_tokens == 0:
                continue
            batch_np = _prepare_sft_batch_np(
                batch_np,
                config=config,
                sft_pack=sft_pack,
                sft_gather_loss=sft_gather_loss,
                sft_bucket_multiple=sft_bucket_multiple,
            )
            loss = _forward_sft_eval_loss(model, batch_np, use_varlen=use_varlen)
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
    eval_microbatch_loss: bool = True,
    defer_final_microbatch_eval: bool = False,
    sft_pack: bool = False,
    sft_gather_loss: bool = False,
    sft_bucket_multiple: int = 128,
    sft_length_bucket_size: int = 2048,
    sft_sampler: str = "sortish",
    async_step_eval: bool = False,
    max_steps: int = 0,
    loss_log_path: str | None = None,
    allow_adamw_fallback: bool = False,
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
    lengths = sequence_lengths(train_dataset)
    if sft_sampler == "homogeneous-step-sorted":
        train_sampler = HomogeneousStepSortedBatchSamplerMLX(
            lengths,
            config.sft_batch_size,
            config.sft_grad_accum_steps,
            bucket_multiple=sft_bucket_multiple,
            drop_last=True,
            seed=0,
        )
    elif sft_sampler == "sorted":
        train_sampler = SortedLengthBatchSamplerMLX(
            lengths, config.sft_batch_size, drop_last=True, seed=0
        )
    else:
        train_sampler = LengthBucketBatchSamplerMLX(
            lengths,
            config.sft_batch_size,
            bucket_size=max(config.sft_batch_size, sft_length_bucket_size),
            drop_last=True,
            seed=0,
        )
    microbatches_per_epoch = (
        len(train_sampler)
        if isinstance(train_sampler, LengthBucketBatchSamplerMLX)
        else len(train_dataset) // config.sft_batch_size
    )
    steps_per_epoch = math.ceil(microbatches_per_epoch / config.sft_grad_accum_steps)
    total_steps = max(1, steps_per_epoch * config.sft_epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False
    reached_max_steps = False
    accum_scale = 1.0 / config.sft_grad_accum_steps
    microbatch_step = _build_microbatch_step(model, accum_scale, compile_step=use_compile)
    varlen_value_and_grad_cache = {}
    monitor_total_steps = min(total_steps, max_steps) if max_steps > 0 else total_steps
    status_writer = TrainingStatusWriter(
        config.checkpoint_dir,
        stage="sft",
        backend="mlx",
        preset=config.preset_name,
        total_steps=monitor_total_steps,
    )
    status_writer.update(force=True, status="running")
    monitor_info = start_background_monitor(status_writer.path, config.checkpoint_dir)
    print(format_monitor_start_message(monitor_info))
    atexit.register(stop_background_monitor, monitor_info)

    def varlen_microbatch_step(batch_mx, cu_key: tuple[int, ...] | None = None):
        x_arg, y_arg, seg_arg, pos_arg, idx_arg, cu_arg = batch_mx[:6]
        if cu_key is None:
            raise RuntimeError("mfa-varlen SFT requires a precomputed static cu-seqlens key")
        static_cu_key = cu_key
        step = varlen_value_and_grad_cache.get(static_cu_key)
        if step is None:
            def loss_fn(model_mod, x_in, y_in, seg_in, pos_in, idx_in):
                _, loss, _ = model_mod(
                    x_in,
                    y_in,
                    return_cache=False,
                    ignore_index=-100,
                    segment_ids=seg_in,
                    position_ids=pos_in,
                    varlen_indices=idx_in,
                    varlen_cu_seqlens=cu_arg,
                )
                return loss * accum_scale

            value_and_grad = nn.value_and_grad(model, loss_fn)
            if use_compile:
                compile_state = [model.state]
                if getattr(model.config, "dropout", 0.0) > 0.0:
                    compile_state.append(mx.random.state)

                @partial(mx.compile, inputs=compile_state, outputs=compile_state)
                def step(x_in, y_in, seg_in, pos_in, idx_in):
                    return value_and_grad(model, x_in, y_in, seg_in, pos_in, idx_in)
            else:
                def step(x_in, y_in, seg_in, pos_in, idx_in):
                    return value_and_grad(model, x_in, y_in, seg_in, pos_in, idx_in)
            varlen_value_and_grad_cache[static_cu_key] = step
        return step(x_arg, y_arg, seg_arg, pos_arg, idx_arg)

    def run_optimizer_step(
        accum_grads,
        accum_loss_lazy: mx.array,
        *,
        micro_in_step: int,
    ) -> tuple[object, mx.array, int, float, float]:
        nonlocal optimizer, global_step, epoch_loss, reached_max_steps

        accum_grads = _scale_partial_accum_grads(accum_grads, config, micro_in_step)
        clipped, _ = (
            (accum_grads, None)
            if config.sft_grad_clip <= 0
            else clip_grads(accum_grads, config.sft_grad_clip)
        )
        progress = global_step / max(total_steps, 1)
        lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (
            1 + math.cos(math.pi * progress)
        )
        optimizer.set_lr(lr)
        try:
            optimizer.update(model, clipped)
        except Exception as exc:
            if should_adamw_fallback(exc, optimizer, config, stage="sft", allow=allow_adamw_fallback):
                print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
                config.sft_optimizer = "adamw"
                print(adamw_fallback_warning("SFT"))
                optimizer = configure_mlx_optimizer(
                    model,
                    config,
                    kind="adamw",
                    learning_rate=config.sft_lr,
                    weight_decay=config.sft_weight_decay,
                )
                optimizer.set_lr(lr)
                optimizer.update(model, clipped)
            else:
                raise
        if async_step_eval:
            mx.async_eval(model.parameters(), optimizer.state_trees())
        else:
            mx.eval(model.parameters(), optimizer.state_trees())

        step_sum_loss = float(accum_loss_lazy.item()) * config.sft_grad_accum_steps
        step_mean_loss = step_sum_loss / max(micro_in_step, 1)
        epoch_loss += step_sum_loss
        global_step += 1
        if loss_log is not None:
            loss_log.write(f"{global_step},{step_mean_loss:.8f},{lr:.12g}\n")
            loss_log.flush()
        if max_steps > 0 and global_step >= max_steps:
            reached_max_steps = True
        status_writer.update(
            status="running",
            step=global_step,
            total_steps=monitor_total_steps,
            train_loss=step_mean_loss,
            lr=lr,
        )
        return None, mx.array(0.0, dtype=mx.float32), 0, step_mean_loss, lr

    profiler = MLXProfile(enabled=profile)
    loss_log = None
    if loss_log_path:
        os.makedirs(os.path.dirname(loss_log_path) or ".", exist_ok=True)
        loss_log = open(loss_log_path, "w", encoding="utf-8")
        loss_log.write("step,loss,lr\n")

    try:
        for epoch in range(config.sft_epochs):
            if reached_max_steps:
                break
            model.train()
            epoch_loss = 0.0
            n_micro = 0
            accum_grads = None
            accum_loss_lazy = mx.array(0.0, dtype=mx.float32)
            micro_in_step = 0

            pbar = tqdm(
                range(microbatches_per_epoch),
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

            def arrays_to_mx_batch(batch_np):
                if profiler.enabled:
                    convert_start = now()
                batch_mx = tuple(mx.array(array) for array in batch_np)
                if profiler.enabled:
                    mx.eval(*batch_mx)
                    profiler.add("numpy_to_mlx", now() - convert_start)
                return batch_mx

            try:
                for _ in pbar:
                    if profiler.enabled:
                        batch_start = now()
                        batch_np = next_batch()
                        profiler.add("batch_fetch", now() - batch_start)
                    else:
                        batch_np = next_batch()
                    if profiler.enabled and sft_pack:
                        pack_start = now()
                    batch_np = _prepare_sft_batch_np(
                        batch_np,
                        config=config,
                        sft_pack=sft_pack,
                        sft_gather_loss=sft_gather_loss,
                        sft_bucket_multiple=sft_bucket_multiple,
                    )
                    if profiler.enabled and sft_pack:
                        profiler.add("sft_pack", now() - pack_start)
                    use_varlen_step = sft_pack and config.attention_backend == "mfa-varlen"
                    varlen_cu_key = (
                        tuple(int(v) for v in batch_np[5])
                        if use_varlen_step and len(batch_np) >= 6
                        else None
                    )
                    batch_mx = arrays_to_mx_batch(batch_np)

                    # Loss-fn has accum_scale baked in.
                    if profiler.enabled:
                        step_start = now()
                        loss, grads = (
                            varlen_microbatch_step(batch_mx, varlen_cu_key)
                            if use_varlen_step
                            else microbatch_step(*batch_mx)
                        )
                        if eval_microbatch_loss:
                            mx.eval(loss, grads)
                        else:
                            mx.eval(grads)
                        profiler.add("forward_backward", now() - step_start)
                    else:
                        loss, grads = (
                            varlen_microbatch_step(batch_mx, varlen_cu_key)
                            if use_varlen_step
                            else microbatch_step(*batch_mx)
                        )
                    accum_grads = _accum_grads(accum_grads, grads)
                    # loss is pre-scaled by accum_scale; sum lazily and only
                    # materialize at end of the optimizer step (or for postfix
                    # every accum_steps microbatches). Avoids forcing a CPU↔GPU
                    # sync between microbatches.
                    accum_loss_lazy = accum_loss_lazy + loss.astype(mx.float32)
                    n_micro += 1
                    micro_in_step += 1
                    should_eval_microbatch = not (
                        defer_final_microbatch_eval
                        and micro_in_step == config.sft_grad_accum_steps
                    )
                    if should_eval_microbatch:
                        eval_target = (
                            (accum_grads, accum_loss_lazy)
                            if eval_microbatch_loss
                            else accum_grads
                        )
                        mx.async_eval(eval_target)

                    if micro_in_step == config.sft_grad_accum_steps:
                        if profiler.enabled:
                            opt_start = now()
                        accum_grads, accum_loss_lazy, micro_in_step, step_mean_loss, _lr = run_optimizer_step(
                            accum_grads,
                            accum_loss_lazy,
                            micro_in_step=micro_in_step,
                        )
                        if profiler.enabled:
                            profiler.add("opt_step", now() - opt_start)
                        pbar.set_postfix(loss=f"{epoch_loss / max(n_micro, 1):.4f}")
                        if reached_max_steps:
                            break
            finally:
                if prefetcher is not None:
                    prefetcher.close()
                pbar.close()

            if micro_in_step > 0:
                if profiler.enabled:
                    opt_start = now()
                accum_grads, accum_loss_lazy, micro_in_step, _step_mean_loss, _lr = run_optimizer_step(
                    accum_grads,
                    accum_loss_lazy,
                    micro_in_step=micro_in_step,
                )
                if profiler.enabled:
                    profiler.add("opt_step", now() - opt_start)

            if reached_max_steps:
                print(f"Reached max SFT optimizer steps: {global_step}")
                if profiler.enabled:
                    print(profiler.format_report(window_label=f"{global_step} optimizer steps"))
                    profiler.reset()
                break

            # Validation
            if profiler.enabled:
                eval_start = now()
                val_loss = _evaluate_sft_loss(
                    model,
                    val_dataset,
                    config.sft_batch_size,
                    config=config,
                    sft_pack=sft_pack,
                    sft_gather_loss=sft_gather_loss,
                    sft_bucket_multiple=sft_bucket_multiple,
                )
                profiler.add("eval", now() - eval_start)
            else:
                val_loss = _evaluate_sft_loss(
                    model,
                    val_dataset,
                    config.sft_batch_size,
                    config=config,
                    sft_pack=sft_pack,
                    sft_gather_loss=sft_gather_loss,
                    sft_bucket_multiple=sft_bucket_multiple,
                )

            print(
                f"Epoch {epoch + 1} | train_loss {epoch_loss / max(n_micro, 1):.4f} | "
                f"val_loss {val_loss:.4f}"
            )
            status_writer.update(
                force=True,
                status="running",
                epoch=epoch + 1,
                epochs=config.sft_epochs,
                step=global_step,
                total_steps=monitor_total_steps,
                train_loss=epoch_loss / max(n_micro, 1),
                val_loss=val_loss,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
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
                        config=config,
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
                        config=config,
                    )
                status_writer.update(
                    force=True,
                    best_val_loss=best_val_loss,
                    best_checkpoint=ckpt_path,
                )
                print(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.sft_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    status_writer.finish(
                        "stopped",
                        message=f"Early stopping at epoch {epoch + 1}",
                        epoch=epoch + 1,
                        epochs=config.sft_epochs,
                        step=global_step,
                        total_steps=monitor_total_steps,
                        val_loss=val_loss,
                        best_val_loss=best_val_loss,
                    )
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
                config=config,
                optimizer=optimizer,
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
                config=config,
                optimizer=optimizer,
            )
        status_writer.finish(
            "interrupted",
            message=f"Saved checkpoint to {interrupt_path}",
            step=global_step,
            total_steps=monitor_total_steps,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        )
        print(f"\nInterrupted during fine-tuning. Saved checkpoint to {interrupt_path}")

    if not interrupted:
        if best_checkpoint_name.endswith("_best.safetensors"):
            final_checkpoint_name = best_checkpoint_name.replace(
                "_best.safetensors", "_final.safetensors"
            )
        else:
            final_checkpoint_name = "sft_final.safetensors"
        final_path = os.path.join(config.checkpoint_dir, final_checkpoint_name)
        _save_sft_checkpoint(
            final_path,
            model,
            meta={
                "step": global_step,
                "best_val_loss": best_val_loss,
                "preset_name": config.preset_name,
                "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                "muon_verified": config.muon_verified,
            },
            config=config,
        )
        print(f"Final SFT checkpoint: {final_path}")

    if interrupted:
        print(f"SFT stopped early. Best val loss so far: {best_val_loss:.4f}")
    elif reached_max_steps:
        status_writer.finish(
            "stopped",
            message=f"Reached max SFT optimizer steps: {global_step}",
            step=global_step,
            total_steps=monitor_total_steps,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        )
        print(f"SFT stopped at max_steps={global_step}. Best val loss so far: {best_val_loss:.4f}")
    else:
        if status_writer.payload.get("status") != "stopped":
            status_writer.finish(
                "complete",
                message="SFT complete",
                step=global_step,
                total_steps=monitor_total_steps,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            )
        print(f"SFT complete. Best val loss: {best_val_loss:.4f}")
    if loss_log is not None:
        loss_log.close()
    if stop_background_monitor(monitor_info):
        print("Monitor stopped.")
    return best_val_loss
