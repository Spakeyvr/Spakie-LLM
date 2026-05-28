"""Benchmark MLX training throughput without checkpoint writes."""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx

from configs.default import get_preset_config
from runtime.mlx_backend import clip_grads, resolve_mlx_runtime
from tokenizer.train_tokenizer import SpakieTokenizer
from training.dataset_mlx import (
    ChatSFTDatasetMLX,
    PackedChatSFTDatasetMLX,
    PretrainDatasetMLX,
    ResumableBatchSamplerMLX,
    stack_batch,
)
from training.mlx_profile import MLXProfile, now
from training.optimizers_mlx import configure_mlx_optimizer
from training.prefetch_mlx import BatchPrefetcher
from training.pretrain_mlx import (
    _accum_grads,
    _arrays_to_mx,
    _build_microbatch_step,
    get_lr,
)
from training.muon_core import OPTIMIZER_CHOICES


def _read_thermal_status() -> str:
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr or "").strip()
    return output.replace("\n", " | ") if output else f"unavailable: exit {result.returncode}"


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


def _load_real_dataset(task: str, config, batch_size: int, *, pack_sft: bool = False):
    if task == "pretrain":
        path = os.path.join(config.processed_data_dir, "train.npy")
        if not os.path.exists(path):
            return None, f"missing pretrain data at {path}"
        dataset = PretrainDatasetMLX(path, config.max_seq_len)
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
    dataset_cls = PackedChatSFTDatasetMLX if pack_sft else ChatSFTDatasetMLX
    dataset = dataset_cls(jsonl_path, tokenizer, config.max_seq_len)
    if len(dataset) < batch_size:
        return None, f"SFT dataset too small for batch_size={batch_size}"
    return dataset, jsonl_path


def _resolve_dataset(task: str, config, batch_size: int, args):
    synthetic_size = max(batch_size * max(args.grad_accum or 1, 1) * 4, 256)
    synthetic = SyntheticSequenceDataset(
        size=synthetic_size,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        ignore_ratio=0.35 if task == "sft" else 0.0,
        seed=0,
    )

    if args.synthetic:
        return synthetic, "synthetic"

    real_dataset, source = _load_real_dataset(
        task,
        config,
        batch_size,
        pack_sft=bool(getattr(args, "pack_sft", False)),
    )
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
    defer_microbatch_eval: bool,
    clear_cache_each_step: bool,
    eval_optimizer_state: bool,
    eval_microbatch_loss: bool,
    capture_random_state: bool,
    async_step_eval: bool,
) -> tuple[float, int, int, float, MLXProfile]:
    profiler = MLXProfile(enabled=True)
    warmup_profiler = MLXProfile(enabled=False)
    sampler = ResumableBatchSamplerMLX(len(dataset), batch_size, drop_last=True, seed=0)
    prefetcher = BatchPrefetcher(dataset, sampler) if use_prefetch else None
    train_iter = None if prefetcher is not None else iter(sampler)
    accum_scale = 1.0 / grad_accum
    microbatch_step = _build_microbatch_step(
        model,
        accum_scale,
        compile_step=use_compile,
        capture_random_state=capture_random_state,
    )

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

    def run_one_step(step_idx: int, active_profiler: MLXProfile) -> tuple[float, int]:
        supervised_count = 0
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)

        for _ in range(grad_accum):
            if active_profiler.enabled:
                batch_start = now()
                x_np, y_np = next_batch()
                active_profiler.add("batch_fetch", now() - batch_start)
            else:
                x_np, y_np = next_batch()
            supervised_count += int(np.count_nonzero(y_np != -100))

            x, y = _arrays_to_mx(x_np, y_np, active_profiler)

            if active_profiler.enabled:
                step_start = now()
                loss, grads = microbatch_step(x, y)
            else:
                loss, grads = microbatch_step(x, y)
            accum_grads = _accum_grads(accum_grads, grads)
            accum_loss = accum_loss + loss.astype(mx.float32)
            if not defer_microbatch_eval:
                if eval_microbatch_loss:
                    mx.eval(accum_grads, accum_loss)
                else:
                    mx.eval(accum_grads)
            if active_profiler.enabled:
                active_profiler.add("forward_backward", now() - step_start)

        if active_profiler.enabled:
            opt_start = now()
            clipped_grads, _ = clip_grads(accum_grads, grad_clip)
        else:
            clipped_grads, _ = clip_grads(accum_grads, grad_clip)
        optimizer.set_lr(lr_fn(step_idx))
        optimizer.update(model, clipped_grads)
        if eval_optimizer_state:
            step_eval_args = (model.parameters(), optimizer.state_trees(), accum_loss)
        else:
            step_eval_args = (model.parameters(), accum_loss)
        if async_step_eval:
            mx.async_eval(*step_eval_args)
        else:
            mx.eval(*step_eval_args)
        if clear_cache_each_step:
            mx.clear_cache()
        loss_value = float(accum_loss.item())
        if active_profiler.enabled:
            active_profiler.add("opt_step", now() - opt_start)
        return loss_value, supervised_count

    last_loss = 0.0
    tokens_processed = 0
    supervised_tokens_processed = 0
    try:
        for step_idx in range(warmup_steps):
            last_loss, _ = run_one_step(step_idx, warmup_profiler)

        profiler.reset()
        start = time.perf_counter()
        for step_idx in range(steps):
            last_loss, supervised_count = run_one_step(step_idx, profiler)
            tokens_processed += batch_size * grad_accum * model.config.max_seq_len
            supervised_tokens_processed += supervised_count
        mx.eval(model.parameters(), optimizer.state_trees())
        elapsed = time.perf_counter() - start
    finally:
        if prefetcher is not None:
            prefetcher.close()

    return elapsed, tokens_processed, supervised_tokens_processed, last_loss, profiler


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLX training throughput without checkpoint writes")
    parser.add_argument("--task", choices=("pretrain", "sft"), default="pretrain", help="Training task to benchmark")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use")
    parser.add_argument("--steps", type=int, default=10, help="Number of timed optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Untimed warmup optimizer steps")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto", help="MLX precision")
    parser.add_argument("--batch-size", type=int, default=0, help="Override task batch size")
    parser.add_argument("--grad-accum", type=int, default=0, help="Override task grad accumulation")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Override grouped-query attention KV heads")
    parser.add_argument("--d-ff", type=int, default=None, help="Override MLP hidden width")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability")
    parser.add_argument("--loss-chunk-size", type=int, default=None, help="Override MLX training loss chunk size")
    parser.add_argument(
        "--mlp-type",
        choices=("gelu", "swiglu"),
        default=None,
        help="Override MLP type",
    )
    parser.add_argument("--swiglu-hidden", type=int, default=None, help="Override SwiGLU hidden width")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default=None,
        help="Optimizer to benchmark (default: task config)",
    )
    parser.add_argument("--muon-ns-steps", type=int, default=None, help="Override Muon Newton-Schulz iteration count")
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
        "--defer-microbatch-eval",
        action="store_true",
        help="Evaluate accumulated gradients only at optimizer-step boundary",
    )
    parser.add_argument(
        "--clear-cache-each-step",
        action="store_true",
        help="Clear the MLX cache after each optimizer step",
    )
    parser.add_argument(
        "--eval-optimizer-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate optimizer state at every optimizer step",
    )
    parser.add_argument(
        "--eval-microbatch-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate accumulated loss at each microbatch boundary",
    )
    parser.add_argument(
        "--capture-random-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Capture MLX random state in the compiled microbatch step",
    )
    parser.add_argument(
        "--async-step-eval",
        action="store_true",
        help="Use mx.async_eval at the optimizer-step boundary and sync once at benchmark end",
    )
    parser.add_argument(
        "--pack-sft",
        action="store_true",
        help="Pack real SFT examples into dense max-length training sequences",
    )
    data_mode = parser.add_mutually_exclusive_group()
    data_mode.add_argument("--synthetic", action="store_true", help="Force synthetic benchmark data")
    data_mode.add_argument("--real-data", action="store_true", help="Use real local datasets when available")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    if args.n_kv_heads is not None:
        config.n_kv_heads = args.n_kv_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.loss_chunk_size is not None:
        config.loss_chunk_size = args.loss_chunk_size
    if args.mlp_type is not None:
        config.mlp_type = args.mlp_type
    if args.swiglu_hidden is not None:
        config.swiglu_hidden = args.swiglu_hidden
    config.refresh_derived_fields()
    capture_random_state = args.capture_random_state
    if capture_random_state is None:
        capture_random_state = config.dropout > 0.0
    if args.optimizer:
        config.pretrain_optimizer = args.optimizer
        config.sft_optimizer = args.optimizer
    if args.muon_ns_steps is not None:
        config.muon_ns_steps = args.muon_ns_steps
    if args.muon_qkv_split is not None:
        config.muon_qkv_split = args.muon_qkv_split
    hparams = _resolve_task_hparams(config, args.task, args)
    batch_size = int(hparams["batch_size"])
    grad_accum = int(hparams["grad_accum"])
    runtime = resolve_mlx_runtime(args.precision)

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
    print(f"Attention KV heads: {config.n_kv_heads or config.n_heads}/{config.n_heads}")
    print(f"MLP: {config.mlp_type}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Optimizer: {optimizer.optimizer_kind}")
    print(f"Compile: {args.compile}")
    print(f"Prefetch: {args.prefetch}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Timed steps: {args.steps}")
    print(f"Thermal before: {_read_thermal_status()}")

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
        elapsed, tokens_processed, supervised_tokens_processed, last_loss, profiler = _benchmark_steps(
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
            defer_microbatch_eval=args.defer_microbatch_eval,
            clear_cache_each_step=args.clear_cache_each_step,
            eval_optimizer_state=args.eval_optimizer_state,
            eval_microbatch_loss=args.eval_microbatch_loss,
            capture_random_state=capture_random_state,
            async_step_eval=args.async_step_eval,
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted before results were available.")
        return
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    step_ms = (elapsed / max(args.steps, 1)) * 1000.0
    iter_per_sec = args.steps / elapsed if elapsed > 0 else 0.0
    tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
    supervised_tok_per_sec = supervised_tokens_processed / elapsed if elapsed > 0 else 0.0
    tokens_per_step = batch_size * grad_accum * config.max_seq_len
    supervised_tokens_per_step = supervised_tokens_processed / max(args.steps, 1)

    print(f"Last loss: {last_loss:.4f}")
    print(f"Avg step: {step_ms:.2f} ms")
    print(f"Iterations/s: {iter_per_sec:.4f}")
    print(f"Tokens/step: {tokens_per_step}")
    print(f"Throughput: {tok_per_sec:.0f} tok/s")
    print(f"Supervised tokens/step: {supervised_tokens_per_step:.1f}")
    print(f"Supervised throughput: {supervised_tok_per_sec:.0f} tok/s")
    print(f"Thermal after: {_read_thermal_status()}")
    print(profiler.format_report(window_label=f"{args.steps} benchmark steps"))


if __name__ == "__main__":
    main()
