"""Compare loss convergence on a learnable synthetic task.

Builds sequences where targets are a deterministic function of inputs
(copy task), giving the model real signal to learn. Compares baseline
(cosine LR + Muon) against variants. Measures train loss at end of fixed
budget and the step at which loss first crosses a threshold.

Same random seeds for model init and data across runs, so the only thing
varying is the algorithmic change under test.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import get_preset_config
from model.transformer_mlx import SpakieGPTMLX
from runtime.mlx_backend import clip_grads, resolve_mlx_runtime
from training.optimizers_mlx import configure_mlx_optimizer
from training.pretrain_mlx import _accum_grads, _build_microbatch_step, get_lr as production_get_lr


def make_copy_dataset(n_examples, seq_len, vocab_size, seed=0):
    """Targets predict the next-input token (copy/shift task).

    For each sequence x of length L sampled from a small-vocab distribution
    with bigram structure, set y[t] = x[t+1]. The model can learn this with
    enough capacity — useful for measuring convergence speed.
    """
    rng = np.random.default_rng(seed)
    # Use a smaller effective vocab so the task is learnable in few steps.
    effective_vocab = min(256, vocab_size)
    # Generate sequences with bigram structure: each token's next token is
    # deterministically (token + 1) mod V, mixed with a small noise rate.
    out_x = np.empty((n_examples, seq_len), dtype=np.int32)
    out_y = np.empty((n_examples, seq_len), dtype=np.int32)
    for i in range(n_examples):
        start = rng.integers(0, effective_vocab)
        seq = (np.arange(seq_len + 1) + start) % effective_vocab
        # Add 5% noise
        noise_mask = rng.random(seq_len + 1) < 0.05
        seq[noise_mask] = rng.integers(0, effective_vocab, size=noise_mask.sum())
        out_x[i] = seq[:-1].astype(np.int32)
        out_y[i] = seq[1:].astype(np.int32)
    return out_x, out_y


def setup_model_and_opt(preset, seed, *, lr, wd, optimizer_kind, ns_steps, perhead, warmup, max_steps):
    config = get_preset_config(preset)
    config.pretrain_optimizer = optimizer_kind
    config.muon_ns_steps = ns_steps
    config.pretrain_lr = lr
    config.pretrain_weight_decay = wd
    config.pretrain_warmup_steps = warmup
    config.pretrain_max_steps = max_steps
    config.dropout = 0.0
    config.muon_qkv_split = bool(perhead)
    runtime = resolve_mlx_runtime("bf16")
    mx.random.seed(seed)
    np.random.seed(seed)
    model = SpakieGPTMLX(config)
    model.set_dtype(runtime.dtype)
    model.train()
    opt = configure_mlx_optimizer(
        model, config, kind=optimizer_kind,
        learning_rate=lr, weight_decay=wd,
    )
    return model, opt, config


def get_lr_cosine(step, config):
    min_lr = config.pretrain_lr * 0.1
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / max(config.pretrain_warmup_steps, 1)
    if step >= config.pretrain_max_steps:
        return min_lr
    progress = (step - config.pretrain_warmup_steps) / (
        config.pretrain_max_steps - config.pretrain_warmup_steps
    )
    return min_lr + 0.5 * (config.pretrain_lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_trapezoid(step, config, decay_frac=0.2):
    """Linear warmup -> constant peak -> linear decay to 10% of peak."""
    min_lr = config.pretrain_lr * 0.1
    decay_start = int(config.pretrain_max_steps * (1 - decay_frac))
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / max(config.pretrain_warmup_steps, 1)
    if step < decay_start:
        return config.pretrain_lr
    if step >= config.pretrain_max_steps:
        return min_lr
    progress = (step - decay_start) / max(config.pretrain_max_steps - decay_start, 1)
    return config.pretrain_lr - progress * (config.pretrain_lr - min_lr)


def run_one(*, label, preset, lr, wd, optimizer_kind, ns_steps, perhead,
            warmup, max_steps, batch_size, dataset, seed, schedule):
    model, opt, config = setup_model_and_opt(
        preset, seed,
        lr=lr, wd=wd, optimizer_kind=optimizer_kind,
        ns_steps=ns_steps, perhead=perhead,
        warmup=warmup, max_steps=max_steps,
    )
    config.pretrain_batch_size = batch_size
    config.pretrain_grad_accum_steps = 1

    accum_scale = 1.0
    microbatch_step = _build_microbatch_step(model, accum_scale, compile_step=True)
    # Use the production get_lr so the eval validates the actually wired-in path.
    config.pretrain_lr_schedule = schedule
    schedule_fn = lambda s: production_get_lr(s, config)

    xs_all, ys_all = dataset
    n = len(xs_all)
    rng = np.random.default_rng(seed)
    losses = []
    t0 = time.perf_counter()
    for step in range(max_steps):
        idx = rng.integers(0, n, size=batch_size)
        x = mx.array(xs_all[idx]); y = mx.array(ys_all[idx])
        loss, grads = microbatch_step(x, y)
        clipped, _ = clip_grads(grads, 1.0)
        opt.set_lr(schedule_fn(step))
        opt.update(model, clipped)
        opt.eval_state()
        mx.eval(model.parameters(), loss)
        losses.append(float(loss.item()))
    elapsed = time.perf_counter() - t0
    final = sum(losses[-10:]) / max(min(10, len(losses)), 1)
    # First step crossing loss=4.0 (signal of having learned something)
    first_below = next((i for i, l in enumerate(losses) if l < 4.0), -1)
    return {
        "label": label,
        "elapsed": elapsed,
        "final_mean_last10": final,
        "first_below_4": first_below,
        "losses": losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="92m")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)  # used only for dataset shape sanity
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    config = get_preset_config(args.preset)
    # Build a learnable dataset (same for all runs).
    xs, ys = make_copy_dataset(
        n_examples=1024, seq_len=config.max_seq_len, vocab_size=config.vocab_size, seed=42,
    )

    variants = [
        dict(label="baseline_cosine_muon", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=5, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_perhead", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=5, perhead=True, warmup=20, schedule="cosine"),
        dict(label="trapezoid_baseline", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=5, perhead=False, warmup=20, schedule="trapezoid"),
        dict(label="cosine_higher_lr", lr=1.2e-3, wd=0.1, optimizer_kind="muon",
             ns_steps=5, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_ns3", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=3, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_ns2", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=2, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_ns2_higher_lr", lr=1.2e-3, wd=0.1, optimizer_kind="muon",
             ns_steps=2, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_ns1", lr=6e-4, wd=0.1, optimizer_kind="muon",
             ns_steps=1, perhead=False, warmup=20, schedule="cosine"),
        dict(label="cosine_ns1_higher_lr", lr=1.2e-3, wd=0.1, optimizer_kind="muon",
             ns_steps=1, perhead=False, warmup=20, schedule="cosine"),
    ]

    print(f"Preset {args.preset} | batch {args.batch_size} | seq {config.max_seq_len} | "
          f"max_steps {args.max_steps} | seed {args.seed}")
    print()
    for v in variants:
        res = run_one(
            **v,
            preset=args.preset, max_steps=args.max_steps,
            batch_size=args.batch_size, dataset=(xs, ys), seed=args.seed,
        )
        ls = res["losses"]
        snap = ", ".join(f"{i:3d}:{ls[i]:5.2f}" for i in [0, 10, 20, 40, min(60, len(ls)-1), len(ls)-1])
        print(f"  {res['label']:25s} | final(last10)={res['final_mean_last10']:.3f} "
              f"| first<4.0 at step {res['first_below_4']:3d} "
              f"| elapsed {res['elapsed']:.1f}s")
        print(f"    {snap}")


if __name__ == "__main__":
    main()
