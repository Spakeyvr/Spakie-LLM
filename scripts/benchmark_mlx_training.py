"""Benchmark MLX training throughput without checkpoint writes."""

from __future__ import annotations

import argparse
import math
import operator
import os
import random
import signal
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn

from configs.default import SUPPORTED_PRESETS, get_preset_config
from runtime.mlx_backend import clip_grads, configure_metal_limits, resolve_mlx_runtime
from tokenizer.train_tokenizer import SpakieTokenizer
from training.dataset_mlx import (
    append_packed_varlen_attention_metadata,
    append_supervised_loss_indices,
    append_valid_token_indices,
    ChatSFTDatasetMLX,
    HomogeneousStepSortedBatchSamplerMLX,
    LengthBucketBatchSamplerMLX,
    PretrainDatasetMLX,
    PretokenizedChatSFTDatasetMLX,
    ResumableBatchSamplerMLX,
    SortedLengthBatchSamplerMLX,
    StepSortedBatchSamplerMLX,
    pack_sft_batch,
    sequence_lengths,
    stack_batch,
    TokenBudgetLengthBatchSamplerMLX,
    trim_after_last_supervised_bucket,
    trim_right_padding_bucket,
    WindowSortedBatchSamplerMLX,
)
from training.mlx_profile import MLXProfile, now
from training.optimizers_mlx import configure_mlx_optimizer
from training.optimizers_mlx import _flatten_arrays, _subset_tree_by_names
from training.prefetch_mlx import BatchPrefetcher
from training.pretrain_mlx import (
    _accum_grads,
    _build_microbatch_step,
    get_lr,
)
from training.muon_core import OPTIMIZER_CHOICES
from mlx.utils import tree_map


class SyntheticSequenceDataset:
    """Small synthetic dataset that matches the MLX dataset contract."""

    def __init__(
        self,
        *,
        size: int,
        seq_len: int,
        vocab_size: int,
        ignore_ratio: float = 0.0,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self._xs = rng.integers(0, vocab_size, size=(size, seq_len), dtype=np.int32)
        self._ys = rng.integers(0, vocab_size, size=(size, seq_len), dtype=np.int32)
        if ignore_ratio > 0.0:
            mask = rng.random((size, seq_len)) < ignore_ratio
            self._ys = self._ys.copy()
            self._ys[mask] = -100

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._xs[idx], self._ys[idx]


def _default_sft_jsonl(config) -> str:
    return os.path.join(config.chat_data_dir, "train.jsonl")


def _resolve_task_hparams(config, task: str, args) -> dict[str, float | int]:
    if task == "pretrain":
        batch_size = args.batch_size or config.pretrain_batch_size
        grad_accum = args.grad_accum or config.pretrain_grad_accum_steps
        config.pretrain_batch_size = batch_size
        config.pretrain_grad_accum_steps = grad_accum
        config.refresh_derived_fields()
        return {
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": config.pretrain_lr,
            "weight_decay": config.pretrain_weight_decay,
            "grad_clip": config.pretrain_grad_clip,
        }

    batch_size = args.batch_size or config.sft_batch_size
    grad_accum = args.grad_accum or config.sft_grad_accum_steps
    config.sft_batch_size = batch_size
    config.sft_grad_accum_steps = grad_accum
    return {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": config.sft_lr,
        "weight_decay": config.sft_weight_decay,
        "grad_clip": config.sft_grad_clip,
    }


def _load_real_dataset(task: str, config, batch_size: int, seq_len: int):
    if task == "pretrain":
        path = os.path.join(config.processed_data_dir, "train.npy")
        if not os.path.exists(path):
            return None, f"missing pretrain data at {path}"
        dataset = PretrainDatasetMLX(path, seq_len)
        if len(dataset) < batch_size:
            return None, f"pretrain dataset too small for batch_size={batch_size}"
        return dataset, path

    jsonl_path = _default_sft_jsonl(config)
    if not os.path.exists(jsonl_path):
        return None, f"missing SFT data at {jsonl_path}"
    tokenizer_path = config.tokenizer_prefix + ".model"
    if not os.path.exists(tokenizer_path):
        return None, f"missing tokenizer at {tokenizer_path}"
    tokenizer = SpakieTokenizer(tokenizer_path)
    dataset = ChatSFTDatasetMLX(jsonl_path, tokenizer, config.max_seq_len)
    if len(dataset) < batch_size:
        return None, f"SFT dataset too small for batch_size={batch_size}"
    return dataset, jsonl_path


def _resolve_dataset(task: str, config, batch_size: int, args):
    seq_len = args.train_seq_len or config.max_seq_len
    synthetic_size = max(batch_size * max(args.grad_accum or 1, 1) * 4, 256)
    synthetic = SyntheticSequenceDataset(
        size=synthetic_size,
        seq_len=seq_len,
        vocab_size=config.vocab_size,
        ignore_ratio=0.35 if task == "sft" else 0.0,
        seed=0,
    )

    if args.synthetic:
        return synthetic, "synthetic"

    real_dataset, source = _load_real_dataset(task, config, batch_size, seq_len)
    if args.real_data:
        if real_dataset is not None:
            return real_dataset, source
        print(f"Real data unavailable ({source}); falling back to synthetic.")
        return synthetic, "synthetic"

    if real_dataset is not None:
        if task == "sft" and args.pretokenize_sft:
            print("Pretokenizing SFT dataset into in-memory NumPy arrays...")
            try:
                real_dataset = PretokenizedChatSFTDatasetMLX(real_dataset)
            except KeyboardInterrupt:
                print("\nInterrupted while pretokenizing SFT dataset.")
                sys.exit(130)
            source = f"{source} (pretokenized)"
        return real_dataset, source
    print(f"Real data unavailable ({source}); using synthetic benchmark data.")
    return synthetic, "synthetic"


def _lr_for_step(task: str, step: int, total_steps: int, config) -> float:
    if task == "pretrain":
        return get_lr(step, config)

    progress = step / max(total_steps, 1)
    return config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (1 + math.cos(math.pi * progress))


def _benchmark_steps(
    *,
    model,
    dataset,
    batch_size: int,
    grad_accum: int,
    lr_fn,
    grad_clip: float,
    optimizer,
    steps: int,
    warmup_steps: int,
    ignore_index: int | None,
    use_compile: bool,
    use_prefetch: bool,
    eval_each_microbatch: bool,
    async_microbatch_eval: bool,
    eval_loss_final_microbatch: bool,
    shapeless_compile: bool,
    compile_accum_step: bool,
    compile_vmap_accum_step: bool,
    compile_vmap_grad_accum_step: bool,
    compile_optimizer_step: bool,
    trim_sft_padding: bool,
    trim_sft_after_last_supervised: bool,
    pack_sft: bool,
    gather_sft_loss: bool,
    sft_bucket_multiple: int,
    compile_full_step: bool,
    full_step_update_repeats: int,
    async_step_eval: bool,
    optimizer_eval_target: str,
    length_bucketed_sft: bool,
    sft_sampler: str,
    sft_length_bucket_size: int,
    sft_token_budget: int,
    sft_window_steps: int,
    eval_microbatch_loss: bool,
    defer_final_microbatch_eval: bool,
    prewarm_sft_shapes: bool,
    loss_log_path: str | None = None,
) -> tuple[float, int, float, MLXProfile, list[float], dict[str, int]]:
    profiler = MLXProfile(enabled=True)
    warmup_profiler = MLXProfile(enabled=False)
    full_step_update_repeats = max(1, int(full_step_update_repeats))
    if compile_vmap_accum_step and compile_vmap_grad_accum_step:
        raise ValueError("Choose only one vmap accumulation mode")
    if sum(
        int(flag)
        for flag in (
            compile_vmap_accum_step,
            compile_vmap_grad_accum_step,
        )
    ) > 1:
        raise ValueError("Choose only one grouped accumulation mode")
    if (
        compile_vmap_accum_step
        or compile_vmap_grad_accum_step
    ) and (
        pack_sft or compile_accum_step or compile_full_step
    ):
        raise ValueError("grouped accumulation is only supported without pack/full/pair accumulation")
    variable_batch_tokens = length_bucketed_sft and sft_sampler == "token-budget"
    if variable_batch_tokens and (
        compile_accum_step
        or compile_full_step
        or compile_vmap_accum_step
        or compile_vmap_grad_accum_step
    ):
        raise ValueError("token-budget SFT sampler currently supports the normal microbatch loop only")
    if length_bucketed_sft and sft_sampler == "token-budget":
        sampler = TokenBudgetLengthBatchSamplerMLX(
            sequence_lengths(dataset),
            token_budget=sft_token_budget or (batch_size * dataset[0][0].shape[0]),
            max_batch_size=max(batch_size * 4, batch_size),
            drop_last=True,
            seed=0,
        )
    elif length_bucketed_sft and sft_sampler == "sorted":
        sampler = SortedLengthBatchSamplerMLX(
            sequence_lengths(dataset),
            batch_size,
            drop_last=True,
            seed=0,
        )
    elif length_bucketed_sft and sft_sampler == "homogeneous-step-sorted":
        sampler = HomogeneousStepSortedBatchSamplerMLX(
            sequence_lengths(dataset),
            batch_size,
            grad_accum,
            bucket_multiple=sft_bucket_multiple,
            drop_last=True,
            seed=0,
        )
    elif length_bucketed_sft and sft_sampler == "step-sorted":
        sampler = StepSortedBatchSamplerMLX(
            sequence_lengths(dataset),
            batch_size,
            grad_accum,
            drop_last=True,
            seed=0,
        )
    elif length_bucketed_sft and sft_sampler == "window-sorted":
        sampler = WindowSortedBatchSamplerMLX(
            sequence_lengths(dataset),
            batch_size,
            grad_accum,
            sft_window_steps,
            drop_last=True,
            seed=0,
        )
    elif length_bucketed_sft:
        sampler = LengthBucketBatchSamplerMLX(
            sequence_lengths(dataset),
            batch_size,
            bucket_size=sft_length_bucket_size,
            drop_last=True,
            seed=0,
        )
    else:
        sampler = ResumableBatchSamplerMLX(len(dataset), batch_size, drop_last=True, seed=0)
    prefetcher = BatchPrefetcher(dataset, sampler) if use_prefetch else None
    train_iter = None if prefetcher is not None else iter(sampler)
    accum_scale = 1.0 / grad_accum
    uses_valid_compaction = (
        getattr(model.config, "compact_valid_mlp", False)
        or getattr(model.config, "compact_valid_projections", False)
    )
    if pack_sft and gather_sft_loss:
        arrays_per_microbatch = 7
    elif pack_sft:
        arrays_per_microbatch = 4
    elif gather_sft_loss and uses_valid_compaction:
        arrays_per_microbatch = 7
    elif uses_valid_compaction:
        arrays_per_microbatch = 4
    elif gather_sft_loss:
        arrays_per_microbatch = 5
    else:
        arrays_per_microbatch = 2
    microbatch_step = _build_microbatch_step(
        model,
        accum_scale,
        compile_step=use_compile,
        ignore_index=ignore_index,
        shapeless=shapeless_compile,
    )
    varlen_value_and_grad_cache = {}

    def varlen_microbatch_step(batch_mx, cu_key: tuple[int, ...] | None = None):
        x_arg, y_arg, seg_arg, pos_arg, idx_arg, cu_arg = batch_mx[:6]
        if use_compile:
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
                        ignore_index=ignore_index,
                        segment_ids=seg_in,
                        position_ids=pos_in,
                        varlen_indices=idx_in,
                        varlen_cu_seqlens=cu_arg,
                    )
                    return loss * accum_scale

                value_and_grad = nn.value_and_grad(model, loss_fn)

                @partial(mx.compile, inputs=[model.state], outputs=[model.state])
                def step(x_in, y_in, seg_in, pos_in, idx_in):
                    return value_and_grad(model, x_in, y_in, seg_in, pos_in, idx_in)

                varlen_value_and_grad_cache[static_cu_key] = step
            return step(x_arg, y_arg, seg_arg, pos_arg, idx_arg)

        def loss_fn(model_mod, x_in, y_in, seg_in, pos_in, idx_in):
            _, loss, _ = model_mod(
                x_in,
                y_in,
                return_cache=False,
                ignore_index=ignore_index,
                segment_ids=seg_in,
                position_ids=pos_in,
                varlen_indices=idx_in,
                varlen_cu_seqlens=cu_arg,
            )
            return loss * accum_scale

        value_and_grad = nn.value_and_grad(model, loss_fn)
        return value_and_grad(model, x_arg, y_arg, seg_arg, pos_arg, idx_arg)

    accum_pair_step = None
    vmap_accum_step = None
    vmap_grad_accum_step = None
    if compile_accum_step:
        if grad_accum < 2:
            raise ValueError("--compile-accum-step requires --grad-accum >= 2")
        loss_and_grad = _build_microbatch_step(
            model, accum_scale, compile_step=False, ignore_index=ignore_index
        )
        state = [model.state]
        if getattr(model.config, "dropout", 0.0) > 0.0:
            state.append(mx.random.state)

        @partial(mx.compile, inputs=state, outputs=state)
        def _pair_step(*arrays):
            loss_sum = mx.array(0.0, dtype=mx.float32)
            grads_sum = None
            for micro_idx in range(grad_accum):
                start = micro_idx * arrays_per_microbatch
                micro = arrays[start : start + arrays_per_microbatch]
                loss, grads = loss_and_grad(*micro)
                loss_sum = loss_sum + loss.astype(mx.float32)
                grads_sum = (
                    grads
                    if grads_sum is None
                    else tree_map(operator.add, grads_sum, grads)
                )
            return loss_sum, grads_sum

        accum_pair_step = _pair_step

    if compile_vmap_accum_step:
        if grad_accum < 2:
            raise ValueError("--compile-vmap-accum-step requires --grad-accum >= 2")

        def _vmap_accum_loss(model_mod, xs, ys):
            def _one_loss(x, y):
                _, loss, _ = model_mod(x, y, return_cache=False, ignore_index=ignore_index)
                return loss

            return mx.vmap(_one_loss)(xs, ys).mean()

        value_and_grad = nn.value_and_grad(model, _vmap_accum_loss)
        state = [model.state]
        if getattr(model.config, "dropout", 0.0) > 0.0:
            state.append(mx.random.state)

        @partial(mx.compile, inputs=state, outputs=state)
        def _vmap_step(xs, ys):
            return value_and_grad(model, xs, ys)

        vmap_accum_step = _vmap_step

    if compile_vmap_grad_accum_step:
        if grad_accum < 2:
            raise ValueError("--compile-vmap-grad-accum-step requires --grad-accum >= 2")

        def _one_loss(model_mod, x, y):
            _, loss, _ = model_mod(x, y, return_cache=False, ignore_index=ignore_index)
            return loss

        one_value_and_grad = nn.value_and_grad(model, _one_loss)

        def _vmap_grad_accum(model_mod, xs, ys):
            def _one(x, y):
                return one_value_and_grad(model_mod, x, y)

            losses, batched_grads = mx.vmap(_one)(xs, ys)
            grads = tree_map(lambda g: g.mean(axis=0), batched_grads)
            return losses.mean(), grads

        state = [model.state]
        if getattr(model.config, "dropout", 0.0) > 0.0:
            state.append(mx.random.state)

        @partial(mx.compile, inputs=state, outputs=state)
        def _vmap_grad_step(xs, ys):
            return _vmap_grad_accum(model, xs, ys)

        vmap_grad_accum_step = _vmap_grad_step

    compiled_optimizer_step = None
    if compile_optimizer_step:
        if hasattr(optimizer, "muon_names"):
            param_flat = _flatten_arrays(model.parameters())
            optimizer.muon_state = {
                name: mx.zeros_like(param_flat[name])
                for name in optimizer.muon_names
                if name in param_flat
            }
            if optimizer.aux_decay_names:
                optimizer.aux_decay.init(
                    _subset_tree_by_names(model.parameters(), set(optimizer.aux_decay_names))
                )
            if optimizer.aux_nodecay_names:
                optimizer.aux_nodecay.init(
                    _subset_tree_by_names(model.parameters(), set(optimizer.aux_nodecay_names))
                )
            opt_state = [
                optimizer.muon_state,
                optimizer.aux_decay.state,
                optimizer.aux_nodecay.state,
            ]
        else:
            optimizer.decay.init(model.parameters())
            optimizer.nodecay.init(model.parameters())
            opt_state = [optimizer.decay.state, optimizer.nodecay.state]

        state = [model.state, *opt_state]

        @partial(mx.compile, inputs=state, outputs=state)
        def _opt_step(grads, loss, lr):
            clipped, _ = (
                (grads, None) if grad_clip <= 0 else clip_grads(grads, grad_clip)
            )
            optimizer.set_lr(lr)
            optimizer.update(model, clipped)
            return loss

        compiled_optimizer_step = _opt_step

    compiled_full_step = None
    if compile_full_step:
        if grad_accum < 2:
            raise ValueError("--compile-full-step requires --grad-accum >= 2")
        loss_and_grad = _build_microbatch_step(
            model, accum_scale, compile_step=False, ignore_index=ignore_index
        )
        if hasattr(optimizer, "muon_names"):
            param_flat = _flatten_arrays(model.parameters())
            optimizer.muon_state = {
                name: mx.zeros_like(param_flat[name])
                for name in optimizer.muon_names
                if name in param_flat
            }
            if optimizer.aux_decay_names:
                optimizer.aux_decay.init(
                    _subset_tree_by_names(model.parameters(), set(optimizer.aux_decay_names))
                )
            if optimizer.aux_nodecay_names:
                optimizer.aux_nodecay.init(
                    _subset_tree_by_names(model.parameters(), set(optimizer.aux_nodecay_names))
                )
            opt_state = [
                optimizer.muon_state,
                optimizer.aux_decay.state,
                optimizer.aux_nodecay.state,
            ]
        else:
            optimizer.decay.init(model.parameters())
            optimizer.nodecay.init(model.parameters())
            opt_state = [optimizer.decay.state, optimizer.nodecay.state]

        state = [model.state, *opt_state]
        if getattr(model.config, "dropout", 0.0) > 0.0:
            state.append(mx.random.state)

        @partial(mx.compile, inputs=state, outputs=state)
        def _full_step(*arrays_and_lr):
            lr_args = arrays_and_lr[-full_step_update_repeats:]
            arrays = arrays_and_lr[:-full_step_update_repeats]
            loss_sum = mx.array(0.0, dtype=mx.float32)
            cursor = 0
            for update_idx in range(full_step_update_repeats):
                grads_sum = None
                for _ in range(grad_accum):
                    micro = arrays[cursor : cursor + arrays_per_microbatch]
                    cursor += arrays_per_microbatch
                    loss, grads = loss_and_grad(*micro)
                    loss_sum = loss_sum + loss.astype(mx.float32)
                    grads_sum = (
                        grads
                        if grads_sum is None
                        else tree_map(operator.add, grads_sum, grads)
                    )
                clipped, _ = (
                    (grads_sum, None)
                    if grad_clip <= 0
                    else clip_grads(grads_sum, grad_clip)
                )
                optimizer.set_lr(lr_args[update_idx])
                optimizer.update(model, clipped)
            return loss_sum

        compiled_full_step = _full_step

    def _arrays_to_mx_batch(batch, active_profiler: MLXProfile):
        if active_profiler.enabled:
            convert_start = now()
        mx_batch = tuple(mx.array(array) for array in batch)
        if active_profiler.enabled:
            mx.eval(*mx_batch)
            active_profiler.add("numpy_to_mlx", now() - convert_start)
        return mx_batch

    def next_batch():
        nonlocal train_iter
        if prefetcher is not None:
            return next(prefetcher)
        try:
            batch_indices = next(train_iter)
        except StopIteration:
            train_iter = iter(sampler)
            batch_indices = next(train_iter)
        return stack_batch(dataset, batch_indices)

    def prepare_batch(batch, active_profiler: MLXProfile):
        x_np, y_np = batch[:2]
        if pack_sft:
            if active_profiler.enabled:
                pack_start = now()
            x_np, y_np = trim_right_padding_bucket(
                x_np, y_np, bucket_multiple=sft_bucket_multiple
            )
            packed = pack_sft_batch(x_np, y_np, max_seq_len=x_np.shape[1])
            if active_profiler.enabled:
                active_profiler.add("sft_pack", now() - pack_start)
            if getattr(model.config, "attention_backend", "sdpa") == "mfa-varlen":
                packed = append_packed_varlen_attention_metadata(packed)
            if gather_sft_loss:
                return append_supervised_loss_indices(
                    packed, bucket_multiple=sft_bucket_multiple
                )
            return packed
        if trim_sft_padding:
            prepared = trim_right_padding_bucket(
                x_np, y_np, bucket_multiple=sft_bucket_multiple
            )
        else:
            prepared = batch
        if trim_sft_after_last_supervised:
            prepared = trim_after_last_supervised_bucket(
                prepared[0],
                prepared[1],
                bucket_multiple=sft_bucket_multiple,
            )
        if model.config.compact_valid_mlp or model.config.compact_valid_projections:
            prepared = append_valid_token_indices(
                prepared, bucket_multiple=sft_bucket_multiple
            )
        if gather_sft_loss:
            return append_supervised_loss_indices(
                prepared, bucket_multiple=sft_bucket_multiple
            )
        return prepared

    token_accounting = {
        "nominal": 0,
        "physical": 0,
        "real": 0,
        "supervised": 0,
    }

    def reset_token_accounting() -> None:
        for key in token_accounting:
            token_accounting[key] = 0

    def account_tokens(original_batch, prepared_batch) -> None:
        x_orig = original_batch[0]
        x = prepared_batch[0]
        token_accounting["nominal"] += int(x_orig.size)
        token_accounting["physical"] += int(x.size)
        if ignore_index is None or len(prepared_batch) < 2:
            token_accounting["real"] += int(x.size)
            token_accounting["supervised"] += int(x.size)
            return

        y = prepared_batch[1]
        token_accounting["supervised"] += int((y != ignore_index).sum())
        if (
            len(prepared_batch) >= 3
            and getattr(prepared_batch[2], "ndim", 0) == 2
            and prepared_batch[2].shape == x.shape
        ):
            token_accounting["real"] += int((prepared_batch[2] >= 0).sum())
        else:
            token_accounting["real"] += int(((x != 0) | (y != ignore_index)).sum())

    def stack_for_vmap(xs: list[mx.array], ys: list[mx.array]) -> tuple[mx.array, mx.array]:
        max_len = max(x.shape[1] for x in xs)
        if all(x.shape[1] == max_len for x in xs):
            return mx.stack(xs), mx.stack(ys)
        padded_xs = []
        padded_ys = []
        target_pad = ignore_index if ignore_index is not None else 0
        for x, y in zip(xs, ys):
            pad_len = max_len - x.shape[1]
            if pad_len <= 0:
                padded_xs.append(x)
                padded_ys.append(y)
                continue
            padded_xs.append(mx.pad(x, [(0, 0), (0, pad_len)], constant_values=0))
            padded_ys.append(mx.pad(y, [(0, 0), (0, pad_len)], constant_values=target_pad))
        return mx.stack(padded_xs), mx.stack(padded_ys)

    def prewarm_compiled_sft_shapes() -> None:
        if ignore_index is None or not prewarm_sft_shapes:
            return
        if compiled_full_step is None and accum_pair_step is None:
            return
        bucket = max(1, sft_bucket_multiple)
        lengths = list(range(bucket, int(dataset[0][0].shape[0]) + 1, bucket))
        for seq_len in lengths:
            x_zero = mx.zeros((batch_size, seq_len), dtype=mx.int32)
            y_ignore = mx.full((batch_size, seq_len), ignore_index, dtype=mx.int32)
            segment_ids = mx.full((batch_size, seq_len), -1, dtype=mx.int32)
            position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32), (batch_size, seq_len))
            valid_indices = mx.zeros((bucket,), dtype=mx.int32)
            valid_mask = mx.zeros((bucket,), dtype=mx.float32)
            loss_indices = mx.zeros((bucket,), dtype=mx.int32)
            loss_targets = mx.zeros((bucket,), dtype=mx.int32)
            loss_mask = mx.zeros((bucket,), dtype=mx.float32)
            arrays = []
            for _ in range(grad_accum):
                if pack_sft and gather_sft_loss:
                    arrays.extend(
                        (
                            x_zero,
                            y_ignore,
                            segment_ids,
                            position_ids,
                            loss_indices,
                            loss_targets,
                            loss_mask,
                        )
                    )
                elif pack_sft:
                    arrays.extend((x_zero, y_ignore, segment_ids, position_ids))
                elif gather_sft_loss and uses_valid_compaction:
                    arrays.extend(
                        (
                            x_zero,
                            y_ignore,
                            valid_indices,
                            valid_mask,
                            loss_indices,
                            loss_targets,
                            loss_mask,
                        )
                    )
                elif uses_valid_compaction:
                    arrays.extend((x_zero, y_ignore, valid_indices, valid_mask))
                elif gather_sft_loss:
                    arrays.extend((x_zero, y_ignore, loss_indices, loss_targets, loss_mask))
                else:
                    arrays.extend((x_zero, y_ignore))
            if compiled_full_step is not None:
                zero_lrs = [
                    mx.array(0.0, dtype=mx.float32)
                    for _ in range(full_step_update_repeats)
                ]
                loss = compiled_full_step(*arrays, *zero_lrs)
                mx.eval(model.parameters(), optimizer.state_trees(), loss)
            elif accum_pair_step is not None:
                loss, grads = accum_pair_step(*arrays)
                mx.eval(grads, loss)

    def run_one_step(step_idx: int, active_profiler: MLXProfile):
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)
        actual_step_tokens = 0

        def eval_optimizer_boundary(loss_value):
            targets = []
            if optimizer_eval_target in {"all", "model"}:
                targets.append(model.parameters())
            if optimizer_eval_target == "all":
                targets.append(optimizer.state_trees())
            targets.append(loss_value)
            if async_step_eval:
                mx.async_eval(*targets)
            else:
                mx.eval(*targets)

        if compiled_full_step is not None:
            arrays = []
            for _ in range(grad_accum * full_step_update_repeats):
                if active_profiler.enabled:
                    batch_start = now()
                    batch_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    batch_np = next_batch()
                prepared_np = prepare_batch(batch_np, active_profiler)
                account_tokens(batch_np, prepared_np)
                if len(prepared_np) != arrays_per_microbatch:
                    raise ValueError(
                        "compiled full step received an unexpected prepared batch "
                        f"with {len(prepared_np)} arrays; expected {arrays_per_microbatch}"
                    )
                arrays.extend(_arrays_to_mx_batch(prepared_np, active_profiler))
            if active_profiler.enabled:
                step_start = now()
            lrs = [
                mx.array(
                    lr_fn(step_idx * full_step_update_repeats + update_idx),
                    dtype=mx.float32,
                )
                for update_idx in range(full_step_update_repeats)
            ]
            accum_loss = compiled_full_step(*arrays, *lrs)
            eval_optimizer_boundary(accum_loss)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)
                active_profiler.add("opt_step", 0.0)
            return accum_loss, actual_step_tokens

        if accum_pair_step is not None:
            arrays = []
            for _ in range(grad_accum):
                if active_profiler.enabled:
                    batch_start = now()
                    batch_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    batch_np = next_batch()
                prepared_np = prepare_batch(batch_np, active_profiler)
                account_tokens(batch_np, prepared_np)
                if len(prepared_np) != arrays_per_microbatch:
                    raise ValueError(
                        "compiled accumulation received an unexpected prepared batch "
                        f"with {len(prepared_np)} arrays; expected {arrays_per_microbatch}"
                    )
                arrays.extend(_arrays_to_mx_batch(prepared_np, active_profiler))
            if active_profiler.enabled:
                step_start = now()
            accum_loss, accum_grads = accum_pair_step(*arrays)
            if async_microbatch_eval:
                mx.async_eval(accum_grads, accum_loss)
            else:
                mx.eval(accum_grads, accum_loss)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)
        elif (
            vmap_accum_step is not None
            or vmap_grad_accum_step is not None
        ):
            xs = []
            ys = []
            for _ in range(grad_accum):
                if active_profiler.enabled:
                    batch_start = now()
                    batch_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    batch_np = next_batch()
                prepared_np = prepare_batch(batch_np, active_profiler)
                account_tokens(batch_np, prepared_np)
                batch_mx = _arrays_to_mx_batch(prepared_np, active_profiler)
                if len(batch_mx) != 2:
                    raise ValueError("grouped accumulation currently supports two-array batches only")
                xs.append(batch_mx[0])
                ys.append(batch_mx[1])
            if active_profiler.enabled:
                step_start = now()
            active_vmap_step = (
                vmap_accum_step
                or vmap_grad_accum_step
            )
            stacked_x, stacked_y = stack_for_vmap(xs, ys)
            accum_loss, accum_grads = active_vmap_step(stacked_x, stacked_y)
            if async_microbatch_eval:
                mx.async_eval(accum_grads, accum_loss)
            else:
                mx.eval(accum_grads, accum_loss)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)
        else:

            for micro_idx in range(grad_accum):
                if active_profiler.enabled:
                    batch_start = now()
                    batch_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    batch_np = next_batch()

                prepared_np = prepare_batch(batch_np, active_profiler)
                account_tokens(batch_np, prepared_np)
                if variable_batch_tokens:
                    actual_step_tokens += int(prepared_np[0].size)
                use_varlen_step = (
                    pack_sft and getattr(model.config, "attention_backend", "sdpa") == "mfa-varlen"
                )
                varlen_cu_key = (
                    tuple(int(v) for v in prepared_np[5])
                    if use_varlen_step and len(prepared_np) >= 6
                    else None
                )
                batch_mx = _arrays_to_mx_batch(prepared_np, active_profiler)

                if active_profiler.enabled:
                    step_start = now()
                    loss, grads = (
                        varlen_microbatch_step(batch_mx, varlen_cu_key)
                        if use_varlen_step
                        else microbatch_step(*batch_mx)
                    )
                else:
                    loss, grads = (
                        varlen_microbatch_step(batch_mx, varlen_cu_key)
                        if use_varlen_step
                        else microbatch_step(*batch_mx)
                    )
                accum_grads = _accum_grads(accum_grads, grads)
                accum_loss = accum_loss + loss.astype(mx.float32)
                should_eval_microbatch = eval_each_microbatch and not (
                    defer_final_microbatch_eval and micro_idx == grad_accum - 1
                )
                if should_eval_microbatch:
                    include_loss = eval_microbatch_loss or (
                        eval_loss_final_microbatch and micro_idx == grad_accum - 1
                    )
                    eval_target = (accum_grads, accum_loss) if include_loss else accum_grads
                    if async_microbatch_eval:
                        mx.async_eval(eval_target)
                    else:
                        mx.eval(eval_target)
                if active_profiler.enabled:
                    active_profiler.add("forward_backward", now() - step_start)

        if active_profiler.enabled:
            opt_start = now()
        lr_float = float(lr_fn(step_idx))
        if compiled_optimizer_step is not None:
            lr = mx.array(lr_float, dtype=mx.float32)
            accum_loss = compiled_optimizer_step(accum_grads, accum_loss, lr)
        else:
            clipped_grads, _ = (
                (accum_grads, None) if grad_clip <= 0 else clip_grads(accum_grads, grad_clip)
            )
            optimizer.set_lr(lr_float)
            optimizer.update(model, clipped_grads)
        eval_optimizer_boundary(accum_loss)
        loss_value = accum_loss
        if active_profiler.enabled:
            active_profiler.add("opt_step", now() - opt_start)
        return loss_value, actual_step_tokens

    last_loss = mx.array(0.0, dtype=mx.float32)
    tokens_processed = 0
    step_tok_per_sec: list[float] = []
    loss_log = None
    prewarm_compiled_sft_shapes()
    if loss_log_path:
        os.makedirs(os.path.dirname(loss_log_path) or ".", exist_ok=True)
        loss_log = open(loss_log_path, "w", encoding="utf-8")
        loss_log.write("step,loss\n")
    try:
        for step_idx in range(warmup_steps):
            last_loss, _ = run_one_step(step_idx, warmup_profiler)

        reset_token_accounting()
        profiler.reset()
        start = time.perf_counter()
        for step_idx in range(steps):
            step_start = time.perf_counter()
            last_loss, actual_step_tokens = run_one_step(step_idx, profiler)
            step_tokens = (
                actual_step_tokens
                if variable_batch_tokens and actual_step_tokens > 0
                else batch_size
                * grad_accum
                * (
                    full_step_update_repeats
                    if compiled_full_step is not None
                    else 1
                )
                * dataset[0][0].shape[0]
            )
            tokens_processed += step_tokens
            step_elapsed = time.perf_counter() - step_start
            if step_elapsed > 0:
                step_tok_per_sec.append(step_tokens / step_elapsed)
            if loss_log is not None:
                loss_log.write(f"{step_idx + 1},{float(last_loss.item()):.8f}\n")
                loss_log.flush()
        mx.eval(model.parameters(), optimizer.state_trees(), last_loss)
        elapsed = time.perf_counter() - start
    finally:
        if loss_log is not None:
            loss_log.close()
        if prefetcher is not None:
            prefetcher.close()

    return (
        elapsed,
        tokens_processed,
        float(last_loss.item()),
        profiler,
        step_tok_per_sec,
        dict(token_accounting),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX training throughput without checkpoint writes")
    parser.add_argument("--task", choices=("pretrain", "sft"), default="pretrain", help="Training task to benchmark")
    parser.add_argument(
        "--preset",
        type=str,
        choices=SUPPORTED_PRESETS,
        default="92m",
        help=f"Model preset to use ({', '.join(SUPPORTED_PRESETS)})",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of timed optimizer steps")
    parser.add_argument(
        "--warmup-steps",
        "--thermal-warmup",
        dest="warmup_steps",
        type=int,
        default=2,
        help="Untimed warmup optimizer steps before timing (also brings the chip to a steady thermal state)",
    )
    parser.add_argument(
        "--loss-log",
        type=str,
        default="",
        help="Optional CSV path for per-timed-step loss; adds a sync and is for parity, not throughput",
    )
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto", help="MLX precision")
    parser.add_argument("--seed", type=int, default=0, help="Seed Python, NumPy, and MLX RNGs")
    parser.add_argument("--mlx-memory-gb", type=float, default=0.0, help="Set MLX Metal memory limit in GB")
    parser.add_argument("--mlx-wired-gb", type=float, default=0.0, help="Set MLX Metal wired memory limit in GB")
    parser.add_argument("--mlx-cache-gb", type=float, default=-1.0, help="Set MLX Metal cache limit in GB (-1 = leave default)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override task batch size")
    parser.add_argument("--grad-accum", type=int, default=0, help="Override task grad accumulation")
    parser.add_argument("--train-seq-len", type=int, default=0, help="Use shorter train sequences without changing model max_seq_len")
    parser.add_argument("--n-layers", type=int, default=None, help="Override number of transformer layers")
    parser.add_argument("--d-model", type=int, default=None, help="Override hidden size")
    parser.add_argument("--n-heads", type=int, default=None, help="Override number of attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Override number of KV attention heads")
    parser.add_argument("--d-ff", type=int, default=None, help="Override FFN hidden size")
    parser.add_argument("--grad-clip", type=float, default=None, help="Override task grad clipping; <=0 disables clipping")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override task learning rate")
    parser.add_argument(
        "--pretrain-warmup-steps",
        type=int,
        default=None,
        help="Override pretrain LR warmup steps",
    )
    parser.add_argument(
        "--sft-lr-warmup-steps",
        type=int,
        default=0,
        help="For SFT benchmarks, linearly ramp LR for this many optimizer steps",
    )
    parser.add_argument(
        "--sft-lr-warmup-start-scale",
        type=float,
        default=1.0,
        help="Starting LR scale for --sft-lr-warmup-steps; 1 keeps the default schedule",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Override model dropout for benchmark comparisons")
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override MLX transformer activation checkpointing for benchmark comparisons",
    )
    parser.add_argument(
        "--mlp-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint only MLP submodules during MLX training",
    )
    parser.add_argument(
        "--compact-valid-mlp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run MLX SFT MLP submodules only on fixed-shape non-padding rows",
    )
    parser.add_argument(
        "--compact-valid-projections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run MLX SFT QKV/out projections only on fixed-shape non-padding rows",
    )
    parser.add_argument(
        "--addmm-residual-projections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fuse residual adds into MLX attention/MLP output projections with mx.addmm",
    )
    parser.add_argument(
        "--mlp-addmm-linears",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use mx.addmm with zero c for biasless MLX MLP linears",
    )
    parser.add_argument(
        "--fused-residual-rmsnorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a custom MLX Metal kernel for residual-add plus RMSNorm in serial RMSNorm blocks",
    )
    parser.add_argument(
        "--contiguous-linear-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force row-contiguous inputs before large MLX linear projections",
    )
    parser.add_argument("--mlp-type", choices=("gelu", "swiglu"), default=None, help="Override MLP block type")
    parser.add_argument("--swiglu-hidden", type=int, default=None, help="Override SwiGLU hidden size")
    parser.add_argument(
        "--qk-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply RMSNorm to per-head Q and K before attention",
    )
    parser.add_argument("--norm-type", choices=("layernorm", "rmsnorm"), default=None, help="Override norm type")
    parser.add_argument("--loss-layout", choices=("flat", "3d", "custom"), default=None, help="Override MLX training loss layout")
    parser.add_argument("--residual-type", choices=("serial", "parallel"), default=None, help="Override residual block type")
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "mfa", "mfa-varlen"),
        default=None,
        help="Override MLX attention backend",
    )
    parser.add_argument(
        "--gelu-variant",
        choices=("exact", "fast"),
        default=None,
        help="Override GELU variant for benchmark comparisons",
    )
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default=None,
        help="Optimizer to benchmark (default: task config)",
    )
    parser.add_argument("--muon-ns-steps", type=int, default=None, help="Override Muon Newton-Schulz iteration count")
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=None,
        help="Override MLX training loss chunk size; 0 disables chunking",
    )
    parser.add_argument(
        "--muon-qkv-split",
        dest="muon_qkv_split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply Muon Newton-Schulz to fused Q/K/V chunks independently",
    )
    parser.add_argument(
        "--grouped-muon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Batch same-shape MLX Muon Newton-Schulz updates across layers",
    )
    parser.add_argument(
        "--compile-muon-ns",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile the per-matrix MLX Muon Newton-Schulz subgraph",
    )
    parser.add_argument(
        "--muon-route",
        choices=("all", "mlp", "attn", "none"),
        default=None,
        help="Benchmark routing for Muon-eligible matrices: all, only MLP, only attention, or none",
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the MLX microbatch step",
    )
    parser.add_argument(
        "--prefetch",
        dest="prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the MLX batch prefetch worker",
    )
    parser.add_argument(
        "--eval-each-microbatch",
        dest="eval_each_microbatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate accumulated grads after each microbatch instead of once per optimizer step",
    )
    parser.add_argument(
        "--async-microbatch-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mx.async_eval for per-microbatch materialization",
    )
    parser.add_argument(
        "--shapeless-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the MLX microbatch step with shapeless=True",
    )
    parser.add_argument(
        "--compile-accum-step",
        action="store_true",
        help="Compile the full two-microbatch gradient accumulation step",
    )
    parser.add_argument(
        "--compile-vmap-accum-step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile a vmap-based gradient accumulation step (default: pretrain preset config)",
    )
    parser.add_argument(
        "--compile-vmap-grad-accum-step",
        action="store_true",
        help="Compile vmap over per-microbatch value_and_grad, then average the gradient tree",
    )
    parser.add_argument(
        "--compile-optimizer-step",
        action="store_true",
        help="Compile gradient clipping and optimizer update",
    )
    parser.add_argument(
        "--compile-full-step",
        action="store_true",
        help="Compile fwd/bwd accumulation plus optimizer update as one graph",
    )
    parser.add_argument(
        "--full-step-update-repeats",
        type=int,
        default=1,
        help="For --compile-full-step, run this many sequential optimizer updates inside one compiled call",
    )
    parser.add_argument(
        "--trim-sft-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim SFT batches to right-padding length buckets",
    )
    parser.add_argument(
        "--sft-trim-after-last-supervised",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trim SFT tokens after the final supervised target in each batch",
    )
    parser.add_argument(
        "--pretokenize-sft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache SFT input/label arrays in memory before benchmarking",
    )
    parser.add_argument(
        "--sft-pack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack logical SFT batches into fewer block-causal physical rows",
    )
    parser.add_argument(
        "--sft-gather-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute SFT vocabulary loss only on supervised target positions",
    )
    parser.add_argument(
        "--async-step-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mx.async_eval after optimizer steps and sync at benchmark boundary",
    )
    parser.add_argument(
        "--optimizer-eval-target",
        choices=("all", "model", "loss"),
        default="all",
        help="Lazy eval target after each optimizer step: all state, model params only, or loss only",
    )
    parser.add_argument(
        "--length-bucketed-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use length-bucketed SFT batches before padding trim",
    )
    parser.add_argument(
        "--sft-sampler",
        choices=(
            "sortish",
            "sorted",
            "homogeneous-step-sorted",
            "step-sorted",
            "window-sorted",
            "token-budget",
        ),
        default="sortish",
        help="SFT length sampler: sortish buckets, globally sorted batches, homogeneous/step/window sorted batches, or static-lattice token-budget batches",
    )
    parser.add_argument(
        "--sft-bucket-multiple",
        type=int,
        default=128,
        help="Round trimmed SFT sequence length up to this multiple",
    )
    parser.add_argument(
        "--sft-length-bucket-size",
        type=int,
        default=2048,
        help="Number of examples per SFT sortish length bucket",
    )
    parser.add_argument(
        "--sft-token-budget",
        type=int,
        default=0,
        help="Dense tokens per microbatch for --sft-sampler token-budget (0 = batch_size * max_seq_len)",
    )
    parser.add_argument(
        "--sft-window-steps",
        type=int,
        default=8,
        help="Optimizer steps per window for --sft-sampler window-sorted",
    )
    parser.add_argument(
        "--eval-microbatch-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize scalar accumulated loss on every microbatch eval",
    )
    parser.add_argument(
        "--eval-loss-final-microbatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --no-eval-microbatch-loss is used, include loss only on final microbatch eval",
    )
    parser.add_argument(
        "--defer-final-microbatch-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip microbatch async_eval on the final microbatch before optimizer sync",
    )
    parser.add_argument(
        "--prewarm-sft-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompile SFT bucket shapes with zero-loss lr=0 dummy batches before warmup",
    )
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--synthetic", action="store_true", help="Force synthetic benchmark data")
    data_mode.add_argument("--real-data", action="store_true", help="Use real local datasets when available")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    config = get_preset_config(args.preset)
    if args.optimizer:
        config.pretrain_optimizer = args.optimizer
        config.sft_optimizer = args.optimizer
    if args.muon_ns_steps is not None:
        config.muon_ns_steps = args.muon_ns_steps
    if args.muon_qkv_split is not None:
        config.muon_qkv_split = args.muon_qkv_split
    if args.grouped_muon is not None:
        config.grouped_muon = args.grouped_muon
    if args.compile_muon_ns is not None:
        config.compile_muon_ns = args.compile_muon_ns
    if args.muon_route is not None:
        config.muon_route = args.muon_route
    if args.loss_chunk_size is not None:
        config.loss_chunk_size = args.loss_chunk_size
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.activation_checkpointing is not None:
        config.activation_checkpointing = args.activation_checkpointing
    if args.mlp_checkpointing is not None:
        config.mlp_checkpointing = args.mlp_checkpointing
    if args.compact_valid_mlp is not None:
        config.compact_valid_mlp = args.compact_valid_mlp
    if args.compact_valid_projections is not None:
        config.compact_valid_projections = args.compact_valid_projections
    if args.addmm_residual_projections is not None:
        config.addmm_residual_projections = args.addmm_residual_projections
    if args.mlp_addmm_linears is not None:
        config.mlp_addmm_linears = args.mlp_addmm_linears
    if args.fused_residual_rmsnorm is not None:
        config.fused_residual_rmsnorm = args.fused_residual_rmsnorm
    if args.contiguous_linear_inputs is not None:
        config.contiguous_linear_inputs = args.contiguous_linear_inputs
    if args.mlp_type is not None:
        config.mlp_type = args.mlp_type
    if args.swiglu_hidden is not None:
        config.swiglu_hidden = args.swiglu_hidden
    if args.norm_type is not None:
        config.norm_type = args.norm_type
    if args.qk_norm is not None:
        config.qk_norm = args.qk_norm
    if args.loss_layout is not None:
        config.loss_layout = args.loss_layout
    if args.residual_type is not None:
        config.residual_type = args.residual_type
    if args.attention_backend is not None:
        config.attention_backend = args.attention_backend
    if args.gelu_variant is not None:
        config.gelu_variant = args.gelu_variant
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.n_kv_heads is not None:
        config.n_kv_heads = args.n_kv_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    config.refresh_derived_fields()
    if args.compile_vmap_accum_step is None:
        args.compile_vmap_accum_step = (
            args.task == "pretrain"
            and bool(getattr(config, "pretrain_vmap_accum_step", False))
        )
    hparams = _resolve_task_hparams(config, args.task, args)
    if args.grad_clip is not None:
        hparams["grad_clip"] = args.grad_clip
    if args.learning_rate is not None:
        hparams["lr"] = args.learning_rate
        if args.task == "pretrain":
            config.pretrain_lr = args.learning_rate
        else:
            config.sft_lr = args.learning_rate
    if args.pretrain_warmup_steps is not None:
        config.pretrain_warmup_steps = args.pretrain_warmup_steps
    batch_size = int(hparams["batch_size"])
    grad_accum = int(hparams["grad_accum"])
    runtime = resolve_mlx_runtime(args.precision)
    configure_metal_limits(
        max_gb=args.mlx_memory_gb if args.mlx_memory_gb > 0 else None,
        wired_gb=args.mlx_wired_gb if args.mlx_wired_gb > 0 else None,
        cache_gb=args.mlx_cache_gb if args.mlx_cache_gb >= 0 else None,
    )

    dataset, data_source = _resolve_dataset(args.task, config, batch_size, args)

    from model.transformer_mlx import SpakieGPTMLX

    model = SpakieGPTMLX(config)
    if runtime.dtype != mx.float32:
        model.set_dtype(runtime.dtype)
    model.train()

    optimizer = configure_mlx_optimizer(
        model,
        config,
        kind=config.pretrain_optimizer if args.task == "pretrain" else config.sft_optimizer,
        learning_rate=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )

    total_steps = max(args.steps * max(1, args.full_step_update_repeats), 1)

    def lr_fn(step: int) -> float:
        lr = _lr_for_step(args.task, step, total_steps, config)
        if args.task == "sft" and args.sft_lr_warmup_steps > 0:
            warmup_steps = max(1, int(args.sft_lr_warmup_steps))
            if step < warmup_steps:
                start_scale = max(0.0, float(args.sft_lr_warmup_start_scale))
                progress = step / warmup_steps
                lr *= start_scale + (1.0 - start_scale) * progress
        return lr

    print(f"Task: {args.task}")
    print(f"Preset: {config.preset_name}")
    print(f"Precision: {runtime.precision}")
    print(f"Seed: {args.seed}")
    print(f"Metal memory limit GB: {args.mlx_memory_gb}")
    print(f"Metal wired limit GB: {args.mlx_wired_gb}")
    print(f"Metal cache limit GB: {args.mlx_cache_gb}")
    print(f"Data: {data_source}")
    print(
        "Shape: "
        f"layers={config.n_layers}, d_model={config.d_model}, "
        f"heads={config.n_heads}, d_ff={config.d_ff}"
    )
    print(f"Attention KV heads: {config.n_kv_heads or config.n_heads}/{config.n_heads}")
    print(f"MLP: {config.mlp_type}")
    print(f"GELU variant: {config.gelu_variant}")
    print(f"Norm type: {config.norm_type}")
    print(f"Loss layout: {config.loss_layout}")
    print(f"Residual type: {config.residual_type}")
    print(f"Attention backend: {config.attention_backend}")
    print(f"Dropout: {config.dropout}")
    print(f"MLP checkpointing: {config.mlp_checkpointing}")
    print(f"Compact valid MLP: {config.compact_valid_mlp}")
    print(f"Compact valid projections: {config.compact_valid_projections}")
    print(f"Addmm residual projections: {config.addmm_residual_projections}")
    print(f"MLP addmm linears: {config.mlp_addmm_linears}")
    print(f"Fused residual RMSNorm: {config.fused_residual_rmsnorm}")
    print(f"Contiguous linear inputs: {config.contiguous_linear_inputs}")
    print(f"Loss chunk size: {config.loss_chunk_size}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Optimizer: {optimizer.optimizer_kind}")
    print(f"Grouped Muon: {config.grouped_muon}")
    print(f"Compile Muon NS: {config.compile_muon_ns}")
    print(f"Muon route: {config.muon_route}")
    print(f"Compile: {args.compile}")
    print(f"Prefetch: {args.prefetch}")
    print(f"Eval each microbatch: {args.eval_each_microbatch}")
    print(f"Async microbatch eval: {args.async_microbatch_eval}")
    print(f"Shapeless compile: {args.shapeless_compile}")
    print(f"Compile accum step: {args.compile_accum_step}")
    print(f"Compile vmap accum step: {args.compile_vmap_accum_step}")
    print(f"Compile vmap grad accum step: {args.compile_vmap_grad_accum_step}")
    print(f"Compile optimizer step: {args.compile_optimizer_step}")
    print(f"Compile full step: {args.compile_full_step}")
    print(f"Full-step update repeats: {args.full_step_update_repeats}")
    print(f"Trim SFT padding: {args.trim_sft_padding}")
    print(f"Trim SFT after last supervised: {args.sft_trim_after_last_supervised}")
    print(f"Pretokenize SFT: {args.pretokenize_sft}")
    print(f"Pack SFT: {args.sft_pack}")
    print(f"Gather SFT loss: {args.sft_gather_loss}")
    print(f"Async step eval: {args.async_step_eval}")
    print(f"Optimizer eval target: {args.optimizer_eval_target}")
    print(f"Length-bucketed SFT: {args.length_bucketed_sft}")
    print(f"SFT sampler: {args.sft_sampler}")
    print(f"SFT bucket multiple: {args.sft_bucket_multiple}")
    print(f"SFT length bucket size: {args.sft_length_bucket_size}")
    print(f"SFT token budget: {args.sft_token_budget}")
    print(f"SFT window steps: {args.sft_window_steps}")
    print(f"SFT LR warmup steps: {args.sft_lr_warmup_steps}")
    print(f"SFT LR warmup start scale: {args.sft_lr_warmup_start_scale}")
    print(f"Eval microbatch loss: {args.eval_microbatch_loss}")
    print(f"Eval loss final microbatch: {args.eval_loss_final_microbatch}")
    print(f"Defer final microbatch eval: {args.defer_final_microbatch_eval}")
    print(f"Prewarm SFT shapes: {args.prewarm_sft_shapes}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Timed steps: {args.steps}")

    previous_sigint_handler = signal.getsignal(signal.SIGINT)
    stop_requested = False

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\nStop requested. Finishing the current benchmark step...")

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        (
            elapsed,
            tokens_processed,
            last_loss,
            profiler,
            step_tok_per_sec,
            token_accounting,
        ) = _benchmark_steps(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr_fn=lr_fn,
            grad_clip=float(hparams["grad_clip"]),
            optimizer=optimizer,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            ignore_index=None if args.task == "pretrain" else -100,
            use_compile=args.compile,
            use_prefetch=args.prefetch,
            eval_each_microbatch=args.eval_each_microbatch,
            async_microbatch_eval=args.async_microbatch_eval,
            eval_loss_final_microbatch=args.eval_loss_final_microbatch,
            shapeless_compile=args.shapeless_compile,
            compile_accum_step=args.compile_accum_step,
            compile_vmap_accum_step=args.compile_vmap_accum_step,
            compile_vmap_grad_accum_step=args.compile_vmap_grad_accum_step,
            compile_optimizer_step=args.compile_optimizer_step,
            trim_sft_padding=args.task == "sft" and args.trim_sft_padding,
            trim_sft_after_last_supervised=(
                args.task == "sft" and args.sft_trim_after_last_supervised
            ),
            pack_sft=args.task == "sft" and args.sft_pack,
            gather_sft_loss=args.task == "sft" and args.sft_gather_loss,
            sft_bucket_multiple=max(1, args.sft_bucket_multiple),
            compile_full_step=args.compile_full_step,
            full_step_update_repeats=args.full_step_update_repeats,
            async_step_eval=args.async_step_eval,
            optimizer_eval_target=args.optimizer_eval_target,
            length_bucketed_sft=args.task == "sft" and args.length_bucketed_sft,
            sft_sampler=args.sft_sampler,
            sft_length_bucket_size=max(
                batch_size, args.sft_length_bucket_size
            ),
            sft_token_budget=args.sft_token_budget,
            sft_window_steps=args.sft_window_steps,
            eval_microbatch_loss=args.eval_microbatch_loss,
            defer_final_microbatch_eval=args.defer_final_microbatch_eval,
            prewarm_sft_shapes=args.task == "sft" and args.prewarm_sft_shapes,
            loss_log_path=args.loss_log or None,
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted before results were available.")
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    step_ms = (elapsed / max(args.steps, 1)) * 1000.0
    tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
    mean_step_tok_per_sec = float(np.mean(step_tok_per_sec)) if step_tok_per_sec else 0.0
    std_step_tok_per_sec = float(np.std(step_tok_per_sec, ddof=1)) if len(step_tok_per_sec) > 1 else 0.0

    print(f"Last loss: {last_loss:.4f}")
    print(f"Avg step: {step_ms:.2f} ms")
    print(f"Throughput: {tok_per_sec:.0f} tok/s")
    print(f"Step throughput mean: {mean_step_tok_per_sec:.0f} tok/s")
    print(f"Step throughput stddev: {std_step_tok_per_sec:.0f} tok/s")
    if elapsed > 0:
        print(
            "Token accounting: "
            f"nominal={token_accounting['nominal']} "
            f"physical={token_accounting['physical']} "
            f"real={token_accounting['real']} "
            f"supervised={token_accounting['supervised']}"
        )
        print(
            "Token accounting rates: "
            f"nominal={token_accounting['nominal'] / elapsed:.0f} tok/s "
            f"physical={token_accounting['physical'] / elapsed:.0f} tok/s "
            f"real={token_accounting['real'] / elapsed:.0f} tok/s "
            f"supervised={token_accounting['supervised'] / elapsed:.0f} tok/s"
        )
        physical = max(token_accounting["physical"], 1)
        nominal = max(token_accounting["nominal"], 1)
        print(
            "Token accounting ratios: "
            f"physical/nominal={token_accounting['physical'] / nominal:.4f} "
            f"real/physical={token_accounting['real'] / physical:.4f} "
            f"supervised/physical={token_accounting['supervised'] / physical:.4f}"
        )
    if len(step_tok_per_sec) >= 600:
        cold = step_tok_per_sec[:300]
        sustained = step_tok_per_sec[-300:]
        print(f"Cold 300-step throughput mean: {float(np.mean(cold)):.0f} tok/s")
        print(f"Cold 300-step throughput stddev: {float(np.std(cold, ddof=1)):.0f} tok/s")
        print(f"Sustained last-300 throughput mean: {float(np.mean(sustained)):.0f} tok/s")
        print(f"Sustained last-300 throughput stddev: {float(np.std(sustained, ddof=1)):.0f} tok/s")
    print(profiler.format_report(window_label=f"{args.steps} benchmark steps"))


if __name__ == "__main__":
    main()
