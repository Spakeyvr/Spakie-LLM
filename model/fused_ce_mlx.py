"""Fused linear cross-entropy for the MLX backend.

The default ``custom`` loss layout computes ``logits = x @ W.T`` and then, in the
backward pass, a full softmax over the (N, vocab) logits followed by
``take_along_axis``/``put_along_axis`` to build ``grad_logits``. At vocab=16384
and N=B*T that softmax+scatter touches the (N, vocab) tensor several times.

``fused_linear_cross_entropy_mean`` keeps the two matmuls (which MLX already runs
on tuned Metal GEMM kernels) but replaces the softmax/scatter chain with a single
Metal kernel that reads the logits once and writes ``grad_logits`` once. Numerics
match the reference path: the loss logsumexp is accumulated in fp32, and
``grad_logits`` is emitted in the logits dtype exactly like the reference.
"""

from __future__ import annotations

import mlx.core as mx


_CE_FWD_KERNELS: dict[tuple[int, int], object] = {}
_CE_BWD_KERNELS: dict[tuple[int, int], object] = {}


def _ce_fwd_kernel(vocab: int, threads: int = 256):
    key = (int(vocab), int(threads))
    kernel = _CE_FWD_KERNELS.get(key)
    if kernel is not None:
        return kernel
    source = f"""
        uint row = thread_position_in_grid.x / {threads}u;
        uint tid = thread_position_in_threadgroup.x;
        constexpr uint V = {int(vocab)}u;
        constexpr uint TH = {int(threads)}u;
        threadgroup float red[TH];
        uint base = row * V;

        float m = -INFINITY;
        for (uint j = tid; j < V; j += TH) m = metal::max(m, float(logits[base + j]));
        red[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TH / 2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] = metal::max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rmax = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint j = tid; j < V; j += TH) acc += metal::exp(float(logits[base + j]) - rmax);
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TH / 2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        if (tid == 0) {{
            float lse = metal::log(red[0]) + rmax;
            uint t = uint(targets[row]);
            loss[row] = lse - float(logits[base + t]);
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"fused_ce_fwd_v{int(vocab)}_t{int(threads)}",
        input_names=["logits", "targets"],
        output_names=["loss"],
        source=source,
        ensure_row_contiguous=True,
    )
    _CE_FWD_KERNELS[key] = kernel
    return kernel


def _ce_bwd_kernel(vocab: int, threads: int = 256):
    key = (int(vocab), int(threads))
    kernel = _CE_BWD_KERNELS.get(key)
    if kernel is not None:
        return kernel
    source = f"""
        uint row = thread_position_in_grid.x / {threads}u;
        uint tid = thread_position_in_threadgroup.x;
        constexpr uint V = {int(vocab)}u;
        constexpr uint TH = {int(threads)}u;
        threadgroup float red[TH];
        uint base = row * V;

        float m = -INFINITY;
        for (uint j = tid; j < V; j += TH) m = metal::max(m, float(logits[base + j]));
        red[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TH / 2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] = metal::max(red[tid], red[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rmax = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint j = tid; j < V; j += TH) acc += metal::exp(float(logits[base + j]) - rmax);
        red[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TH / 2; s > 0; s >>= 1) {{
            if (tid < s) red[tid] += red[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float inv = 1.0f / red[0];
        float sc = scale[0];
        uint t = uint(targets[row]);
        for (uint j = tid; j < V; j += TH) {{
            float p = metal::exp(float(logits[base + j]) - rmax) * inv;
            if (j == t) p -= 1.0f;
            grad[base + j] = T(p * sc);
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"fused_ce_bwd_v{int(vocab)}_t{int(threads)}",
        input_names=["logits", "targets", "scale"],
        output_names=["grad"],
        source=source,
        ensure_row_contiguous=True,
    )
    _CE_BWD_KERNELS[key] = kernel
    return kernel


def _fwd_loss(logits: mx.array, targets: mx.array) -> mx.array:
    n, vocab = logits.shape
    threads = 256
    (loss_rows,) = _ce_fwd_kernel(vocab, threads)(
        inputs=[logits, targets],
        grid=(n * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return loss_rows.mean()


def _bwd_grad_logits(logits: mx.array, targets: mx.array, scale: mx.array) -> mx.array:
    n, vocab = logits.shape
    threads = 256
    (grad,) = _ce_bwd_kernel(vocab, threads)(
        inputs=[logits, targets, scale.reshape(1)],
        template=[("T", logits.dtype)],
        grid=(n * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(n, vocab)],
        output_dtypes=[logits.dtype],
    )
    return grad


@mx.custom_function
def fused_linear_cross_entropy_mean(
    flat_x: mx.array, weight: mx.array, targets: mx.array
) -> mx.array:
    logits = flat_x @ weight.T
    return _fwd_loss(logits, targets)


@fused_linear_cross_entropy_mean.vjp
def _fused_linear_cross_entropy_mean_vjp(primals, cotangent, output):
    flat_x, weight, targets = primals
    logits = flat_x @ weight.T
    n = flat_x.shape[0]
    scale = cotangent / n
    grad_logits = _bwd_grad_logits(logits, targets, scale)
    grad_x = grad_logits @ weight
    grad_weight = grad_logits.T @ flat_x
    return grad_x, grad_weight, mx.zeros_like(targets)
