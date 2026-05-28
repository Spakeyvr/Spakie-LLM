"""Benchmark MLX training throughput without checkpoint writes."""

from __future__ import annotations

import argparse
import math
import operator
import os
import signal
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from configs.default import get_preset_config
from runtime.mlx_backend import clip_grads, configure_metal_limits, resolve_mlx_runtime
from tokenizer.train_tokenizer import SpakieTokenizer
from training.dataset_mlx import (
    ChatSFTDatasetMLX,
    LengthBucketBatchSamplerMLX,
    PretrainDatasetMLX,
    ResumableBatchSamplerMLX,
    sequence_lengths,
    stack_batch,
    trim_right_padding_bucket,
)
from training.mlx_profile import MLXProfile, now
from training.optimizers_mlx import configure_mlx_optimizer
from training.optimizers_mlx import _flatten_arrays, _subset_tree_by_names
from training.prefetch_mlx import BatchPrefetcher
from training.pretrain_mlx import (
    _accum_grads,
    _arrays_to_mx,
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
    use_compile: bool,
    use_prefetch: bool,
    eval_each_microbatch: bool,
    async_microbatch_eval: bool,
    shapeless_compile: bool,
    compile_accum_step: bool,
    compile_optimizer_step: bool,
    trim_sft_padding: bool,
    compile_full_step: bool,
    async_step_eval: bool,
    length_bucketed_sft: bool,
    eval_microbatch_loss: bool,
) -> tuple[float, int, float, MLXProfile]:
    profiler = MLXProfile(enabled=True)
    warmup_profiler = MLXProfile(enabled=False)
    if length_bucketed_sft:
        sampler = LengthBucketBatchSamplerMLX(
            sequence_lengths(dataset), batch_size, drop_last=True, seed=0
        )
    else:
        sampler = ResumableBatchSamplerMLX(len(dataset), batch_size, drop_last=True, seed=0)
    prefetcher = BatchPrefetcher(dataset, sampler) if use_prefetch else None
    train_iter = None if prefetcher is not None else iter(sampler)
    accum_scale = 1.0 / grad_accum
    microbatch_step = _build_microbatch_step(
        model, accum_scale, compile_step=use_compile, shapeless=shapeless_compile
    )
    accum_pair_step = None
    if compile_accum_step:
        if grad_accum != 2:
            raise ValueError("--compile-accum-step currently requires --grad-accum 2")
        loss_and_grad = _build_microbatch_step(
            model, accum_scale, compile_step=False
        )
        state = [model.state]
        if getattr(model.config, "dropout", 0.0) > 0.0:
            state.append(mx.random.state)

        @partial(mx.compile, inputs=state, outputs=state)
        def _pair_step(x0, y0, x1, y1):
            loss0, grads0 = loss_and_grad(x0, y0)
            loss1, grads1 = loss_and_grad(x1, y1)
            grads = tree_map(operator.add, grads0, grads1)
            return loss0 + loss1, grads

        accum_pair_step = _pair_step

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
            clipped, _ = (grads, None) if grad_clip <= 0 else clip_grads(grads, grad_clip)
            optimizer.set_lr(lr)
            optimizer.update(model, clipped)
            return loss

        compiled_optimizer_step = _opt_step

    compiled_full_step = None
    if compile_full_step:
        if grad_accum != 2:
            raise ValueError("--compile-full-step currently requires --grad-accum 2")
        loss_and_grad = _build_microbatch_step(
            model, accum_scale, compile_step=False
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
        def _full_step(x0, y0, x1, y1, lr):
            loss0, grads0 = loss_and_grad(x0, y0)
            loss1, grads1 = loss_and_grad(x1, y1)
            grads = tree_map(operator.add, grads0, grads1)
            clipped, _ = (grads, None) if grad_clip <= 0 else clip_grads(grads, grad_clip)
            optimizer.set_lr(lr)
            optimizer.update(model, clipped)
            return loss0 + loss1

        compiled_full_step = _full_step

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

    def run_one_step(step_idx: int, active_profiler: MLXProfile):
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)

        if compiled_full_step is not None:
            arrays = []
            for _ in range(2):
                if active_profiler.enabled:
                    batch_start = now()
                    x_np, y_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    x_np, y_np = next_batch()
                if trim_sft_padding:
                    x_np, y_np = trim_right_padding_bucket(x_np, y_np)
                arrays.extend(_arrays_to_mx(x_np, y_np, active_profiler))
            if active_profiler.enabled:
                step_start = now()
            accum_loss = compiled_full_step(
                *arrays, mx.array(lr_fn(step_idx), dtype=mx.float32)
            )
            if async_step_eval:
                mx.async_eval(model.parameters(), optimizer.state_trees(), accum_loss)
            else:
                mx.eval(model.parameters(), optimizer.state_trees(), accum_loss)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)
                active_profiler.add("opt_step", 0.0)
            return accum_loss

        if accum_pair_step is not None:
            arrays = []
            for _ in range(2):
                if active_profiler.enabled:
                    batch_start = now()
                    x_np, y_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    x_np, y_np = next_batch()
                if trim_sft_padding:
                    x_np, y_np = trim_right_padding_bucket(x_np, y_np)
                arrays.extend(_arrays_to_mx(x_np, y_np, active_profiler))
            if active_profiler.enabled:
                step_start = now()
            accum_loss, accum_grads = accum_pair_step(*arrays)
            if async_microbatch_eval:
                mx.async_eval(accum_grads, accum_loss)
            else:
                mx.eval(accum_grads, accum_loss)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)
        else:

            for _ in range(grad_accum):
                if active_profiler.enabled:
                    batch_start = now()
                    x_np, y_np = next_batch()
                    active_profiler.add("batch_fetch", now() - batch_start)
                else:
                    x_np, y_np = next_batch()
                if trim_sft_padding:
                    x_np, y_np = trim_right_padding_bucket(x_np, y_np)

                x, y = _arrays_to_mx(x_np, y_np, active_profiler)

                if active_profiler.enabled:
                    step_start = now()
                    loss, grads = microbatch_step(x, y)
                else:
                    loss, grads = microbatch_step(x, y)
                accum_grads = _accum_grads(accum_grads, grads)
                accum_loss = accum_loss + loss.astype(mx.float32)
                if eval_each_microbatch:
                    eval_target = (accum_grads, accum_loss) if eval_microbatch_loss else accum_grads
                    if async_microbatch_eval:
                        mx.async_eval(eval_target)
                    else:
                        mx.eval(eval_target)
                if active_profiler.enabled:
                    active_profiler.add("forward_backward", now() - step_start)

        if active_profiler.enabled:
            opt_start = now()
        lr = mx.array(lr_fn(step_idx), dtype=mx.float32)
        if compiled_optimizer_step is not None:
            accum_loss = compiled_optimizer_step(accum_grads, accum_loss, lr)
        else:
            clipped_grads, _ = (
                (accum_grads, None) if grad_clip <= 0 else clip_grads(accum_grads, grad_clip)
            )
            optimizer.set_lr(float(lr.item()))
            optimizer.update(model, clipped_grads)
        if async_step_eval:
            mx.async_eval(model.parameters(), optimizer.state_trees(), accum_loss)
        else:
            mx.eval(model.parameters(), optimizer.state_trees(), accum_loss)
        loss_value = accum_loss
        if active_profiler.enabled:
            active_profiler.add("opt_step", now() - opt_start)
        return loss_value

    last_loss = mx.array(0.0, dtype=mx.float32)
    tokens_processed = 0
    try:
        for step_idx in range(warmup_steps):
            last_loss = run_one_step(step_idx, warmup_profiler)

        profiler.reset()
        start = time.perf_counter()
        for step_idx in range(steps):
            last_loss = run_one_step(step_idx, profiler)
            tokens_processed += batch_size * grad_accum * dataset[0][0].shape[0]
        mx.eval(model.parameters(), optimizer.state_trees(), last_loss)
        elapsed = time.perf_counter() - start
    finally:
        if prefetcher is not None:
            prefetcher.close()

    return elapsed, tokens_processed, float(last_loss.item()), profiler


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX training throughput without checkpoint writes")
    parser.add_argument("--task", choices=("pretrain", "sft"), default="pretrain", help="Training task to benchmark")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Untimed warmup optimizer steps")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto", help="MLX precision")
    parser.add_argument("--mlx-memory-gb", type=float, default=0.0, help="Set MLX Metal memory limit in GB")
    parser.add_argument("--mlx-wired-gb", type=float, default=0.0, help="Set MLX Metal wired memory limit in GB")
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
    parser.add_argument("--dropout", type=float, default=None, help="Override model dropout for benchmark comparisons")
    parser.add_argument("--mlp-type", choices=("gelu", "swiglu"), default=None, help="Override MLP block type")
    parser.add_argument("--swiglu-hidden", type=int, default=None, help="Override SwiGLU hidden size")
    parser.add_argument("--norm-type", choices=("layernorm", "rmsnorm"), default=None, help="Override norm type")
    parser.add_argument("--loss-layout", choices=("flat", "3d", "custom"), default=None, help="Override MLX training loss layout")
    parser.add_argument("--residual-type", choices=("serial", "parallel"), default=None, help="Override residual block type")
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
        "--trim-sft-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim SFT batches to right-padding length buckets",
    )
    parser.add_argument(
        "--async-step-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mx.async_eval after optimizer steps and sync at benchmark boundary",
    )
    parser.add_argument(
        "--length-bucketed-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use length-bucketed SFT batches before padding trim",
    )
    parser.add_argument(
        "--eval-microbatch-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Materialize scalar accumulated loss on every microbatch eval",
    )
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--synthetic", action="store_true", help="Force synthetic benchmark data")
    data_mode.add_argument("--real-data", action="store_true", help="Use real local datasets when available")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    if args.optimizer:
        config.pretrain_optimizer = args.optimizer
        config.sft_optimizer = args.optimizer
    if args.muon_ns_steps is not None:
        config.muon_ns_steps = args.muon_ns_steps
    if args.muon_qkv_split is not None:
        config.muon_qkv_split = args.muon_qkv_split
    if args.loss_chunk_size is not None:
        config.loss_chunk_size = args.loss_chunk_size
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.mlp_type is not None:
        config.mlp_type = args.mlp_type
    if args.swiglu_hidden is not None:
        config.swiglu_hidden = args.swiglu_hidden
    if args.norm_type is not None:
        config.norm_type = args.norm_type
    if args.loss_layout is not None:
        config.loss_layout = args.loss_layout
    if args.residual_type is not None:
        config.residual_type = args.residual_type
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
    hparams = _resolve_task_hparams(config, args.task, args)
    if args.grad_clip is not None:
        hparams["grad_clip"] = args.grad_clip
    if args.learning_rate is not None:
        hparams["lr"] = args.learning_rate
        if args.task == "pretrain":
            config.pretrain_lr = args.learning_rate
        else:
            config.sft_lr = args.learning_rate
    batch_size = int(hparams["batch_size"])
    grad_accum = int(hparams["grad_accum"])
    runtime = resolve_mlx_runtime(args.precision)
    configure_metal_limits(
        max_gb=args.mlx_memory_gb if args.mlx_memory_gb > 0 else None,
        wired_gb=args.mlx_wired_gb if args.mlx_wired_gb > 0 else None,
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

    total_steps = max(args.steps, 1)
    lr_fn = lambda step: _lr_for_step(args.task, step, total_steps, config)

    print(f"Task: {args.task}")
    print(f"Preset: {config.preset_name}")
    print(f"Precision: {runtime.precision}")
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
    print(f"Dropout: {config.dropout}")
    print(f"Loss chunk size: {config.loss_chunk_size}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Optimizer: {optimizer.optimizer_kind}")
    print(f"Compile: {args.compile}")
    print(f"Prefetch: {args.prefetch}")
    print(f"Eval each microbatch: {args.eval_each_microbatch}")
    print(f"Async microbatch eval: {args.async_microbatch_eval}")
    print(f"Shapeless compile: {args.shapeless_compile}")
    print(f"Compile accum step: {args.compile_accum_step}")
    print(f"Compile optimizer step: {args.compile_optimizer_step}")
    print(f"Compile full step: {args.compile_full_step}")
    print(f"Trim SFT padding: {args.trim_sft_padding}")
    print(f"Async step eval: {args.async_step_eval}")
    print(f"Length-bucketed SFT: {args.length_bucketed_sft}")
    print(f"Eval microbatch loss: {args.eval_microbatch_loss}")
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
        elapsed, tokens_processed, last_loss, profiler = _benchmark_steps(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            grad_accum=grad_accum,
            lr_fn=lr_fn,
            grad_clip=float(hparams["grad_clip"]),
            optimizer=optimizer,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            use_compile=args.compile,
            use_prefetch=args.prefetch,
            eval_each_microbatch=args.eval_each_microbatch,
            async_microbatch_eval=args.async_microbatch_eval,
            shapeless_compile=args.shapeless_compile,
            compile_accum_step=args.compile_accum_step,
            compile_optimizer_step=args.compile_optimizer_step,
            trim_sft_padding=args.task == "sft" and args.trim_sft_padding,
            compile_full_step=args.compile_full_step,
            async_step_eval=args.async_step_eval,
            length_bucketed_sft=args.task == "sft" and args.length_bucketed_sft,
            eval_microbatch_loss=args.eval_microbatch_loss,
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted before results were available.")
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    step_ms = (elapsed / max(args.steps, 1)) * 1000.0
    tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0

    print(f"Last loss: {last_loss:.4f}")
    print(f"Avg step: {step_ms:.2f} ms")
    print(f"Throughput: {tok_per_sec:.0f} tok/s")
    print(profiler.format_report(window_label=f"{args.steps} benchmark steps"))


if __name__ == "__main__":
    main()
