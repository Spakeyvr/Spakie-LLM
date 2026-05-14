"""Compare full pretrain step time with dropout=0.1 vs dropout=0."""

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


def bench_preset(preset: str, dropout: float, iters: int = 5):
    config = get_preset_config(preset)
    config.dropout = dropout
    model = SpakieGPTMLX(config)
    model.set_dtype(mx.bfloat16)
    model.train()

    B = config.pretrain_batch_size
    T = config.max_seq_len
    K = config.pretrain_grad_accum_steps
    x = mx.random.randint(0, config.vocab_size, (B, T))
    y = mx.random.randint(0, config.vocab_size, (B, T))
    mx.eval(x, y, model.parameters())

    def loss_fn(model, x, y):
        _, loss, _ = model(x, y, return_cache=False, ignore_index=None)
        return loss * (1.0 / K)

    vag = nn.value_and_grad(model, loss_fn)
    state = [model.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        return vag(model, x, y)

    def one_step():
        accum_grads = None
        accum_loss = mx.array(0.0, dtype=mx.float32)
        for _ in range(K):
            loss, grads = step(x, y)
            accum_grads = grads if accum_grads is None else tree_map(operator.add, accum_grads, grads)
            accum_loss = accum_loss + loss.astype(mx.float32)
        mx.eval(accum_grads, accum_loss)

    for _ in range(2):
        one_step()

    t0 = time.perf_counter()
    for _ in range(iters):
        one_step()
    dt = (time.perf_counter() - t0) / iters
    tokens = B * T * K
    print(f"{preset:5s} dropout={dropout:.1f} | {dt*1000:7.0f} ms/step | {tokens/dt:7.0f} tok/s")
    return dt


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "92m"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    d1 = bench_preset(preset, 0.1, iters)
    d0 = bench_preset(preset, 0.0, iters)
    print(f"speedup (dropout 0.1 -> 0.0): {d1/d0:.2f}x")


if __name__ == "__main__":
    main()
