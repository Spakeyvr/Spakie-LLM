"""A/B benchmark: per-microbatch eval vs deferred eval, across presets.

Mirrors the pretrain inner loop closely enough to compare the two strategies.
"""

import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import operator
from mlx.utils import tree_map

from configs.default import get_preset_config
from model.transformer_mlx import SpakieGPTMLX


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "180m"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    config = get_preset_config(preset)
    model = SpakieGPTMLX(config)
    model.set_dtype(mx.bfloat16)
    model.train()

    B = config.pretrain_batch_size
    T = config.max_seq_len
    K = config.pretrain_grad_accum_steps
    print(
        f"Preset={preset} B={B} T={T} K={K} "
        f"d_model={config.d_model} layers={config.n_layers}"
    )
    x = mx.random.randint(0, config.vocab_size, (B, T))
    y = mx.random.randint(0, config.vocab_size, (B, T))
    mx.eval(x, y, model.parameters())

    def loss_fn(model, x, y):
        _, loss, _ = model(x, y, return_cache=False)
        return loss * (1.0 / K)

    vag = nn.value_and_grad(model, loss_fn)
    state = [model.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        return vag(model, x, y)

    def accum_and_eval_per_microbatch():
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)
        for _ in range(K):
            loss, grads = step(x, y)
            accum_grads = grads if accum_grads is None else tree_map(operator.add, accum_grads, grads)
            accum_loss = accum_loss + loss.astype(mx.float32)
            mx.eval(accum_grads, accum_loss)  # per-microbatch sync
        mx.eval(accum_grads, accum_loss)

    def accum_and_eval_deferred():
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)
        for _ in range(K):
            loss, grads = step(x, y)
            accum_grads = grads if accum_grads is None else tree_map(operator.add, accum_grads, grads)
            accum_loss = accum_loss + loss.astype(mx.float32)
        mx.eval(accum_grads, accum_loss)  # single sync at end

    # Warmup once (compiles the function).
    for _ in range(2):
        accum_and_eval_per_microbatch()
        accum_and_eval_deferred()

    def time_fn(label, fn):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters
        tokens = B * T * K
        print(f"{label:32s} | {dt*1000:8.1f} ms/step | {tokens/dt:7.0f} tok/s")
        return dt

    a = time_fn("per-microbatch eval", accum_and_eval_per_microbatch)
    b = time_fn("deferred eval (new)", accum_and_eval_deferred)
    print(f"speedup: {a/b:.2f}x")


if __name__ == "__main__":
    main()
