"""MLX mirror of SpakieGPT. Uses mx.fast.scaled_dot_product_attention for the Metal kernel."""

from __future__ import annotations

import math
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.utils as nn_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


_MFA_FLASH_ATTENTION = None


def _mfa_flash_attention():
    global _MFA_FLASH_ATTENTION
    if _MFA_FLASH_ATTENTION is None:
        try:
            from mlx_mfa import flash_attention
        except ImportError as exc:
            raise RuntimeError(
                "attention_backend='mfa' requires the optional mlx-mfa package"
            ) from exc
        _MFA_FLASH_ATTENTION = flash_attention
    return _MFA_FLASH_ATTENTION


_MFA_FLASH_ATTENTION_VARLEN = None


def _mfa_flash_attention_varlen():
    global _MFA_FLASH_ATTENTION_VARLEN
    if _MFA_FLASH_ATTENTION_VARLEN is None:
        try:
            from mlx_mfa import flash_attention_varlen
        except ImportError as exc:
            raise RuntimeError(
                "attention_backend='mfa-varlen' requires the optional mlx-mfa package"
            ) from exc
        _MFA_FLASH_ATTENTION_VARLEN = flash_attention_varlen
    return _MFA_FLASH_ATTENTION_VARLEN


_MFA_FLASH_ATTENTION_VARLEN_QKV_PACKED = None


def _mfa_flash_attention_varlen_qkv_packed():
    global _MFA_FLASH_ATTENTION_VARLEN_QKV_PACKED
    if _MFA_FLASH_ATTENTION_VARLEN_QKV_PACKED is None:
        try:
            from mlx_mfa import flash_attention_varlen_qkv_packed
        except ImportError as exc:
            raise RuntimeError(
                "attention_backend='mfa-varlen' requires the optional mlx-mfa package"
            ) from exc
        _MFA_FLASH_ATTENTION_VARLEN_QKV_PACKED = flash_attention_varlen_qkv_packed
    return _MFA_FLASH_ATTENTION_VARLEN_QKV_PACKED


def _gelu_exact(x: mx.array) -> mx.array:
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2


def _gelu_fast(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


def _lower_right_causal_mask(query_length: int, key_length: int) -> mx.array:
    """Boolean causal mask aligned to the lower-right of a rectangular score matrix."""
    offset = key_length - query_length
    return mx.arange(key_length)[None, :] <= (mx.arange(query_length)[:, None] + offset)


def _attention_with_dropout(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: mx.array | None,
    dropout: nn.Dropout,
) -> mx.array:
    """Reference SDPA path used when attention-weight dropout is active."""
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = mx.repeat(k, repeat, axis=1)
        v = mx.repeat(v, repeat, axis=1)
    scores = (q * scale) @ k.swapaxes(-1, -2)
    if mask is None:
        mask = _lower_right_causal_mask(q.shape[-2], k.shape[-2])
    if mask.dtype == mx.bool_:
        valid_rows = mx.any(mask, axis=-1, keepdims=True)
        scores = mx.where(
            mask,
            scores,
            mx.array(mx.finfo(scores.dtype).min, dtype=scores.dtype),
        )
    else:
        scores = scores + mask
    weights = dropout(mx.softmax(scores, axis=-1))
    if mask.dtype == mx.bool_:
        # Packed batches contain padded queries with no valid keys. SDPA emits
        # zero for those rows; make the reference dropout path equally finite.
        weights = mx.where(valid_rows, weights, mx.zeros_like(weights))
    return weights @ v


@mx.custom_function
def _linear_cross_entropy_mean(flat_x: mx.array, weight: mx.array, targets: mx.array) -> mx.array:
    logits = flat_x @ weight.T
    if logits.dtype == mx.float16:
        logits = logits.astype(mx.float32)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


@_linear_cross_entropy_mean.vjp
def _linear_cross_entropy_mean_vjp(primals, cotangent, output):
    flat_x, weight, targets = primals
    logits = flat_x @ weight.T
    if logits.dtype == mx.float16:
        logits = logits.astype(mx.float32)
    probs = mx.softmax(logits, axis=-1)
    target_idx = targets[:, None]
    target_updates = mx.take_along_axis(probs, target_idx, axis=1) - 1
    grad_logits = mx.put_along_axis(probs, target_idx, target_updates, axis=1) / flat_x.shape[0]
    grad_logits = grad_logits * cotangent
    grad_x = grad_logits @ weight
    grad_weight = grad_logits.T @ flat_x
    return grad_x, grad_weight, mx.zeros_like(targets)


class CausalSelfAttentionMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if self.n_kv_heads == self.n_heads:
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
            self.q_proj = None
            self.kv_proj = None
        else:
            self.qkv = None
            self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.kv_proj = nn.Linear(
                config.d_model, 2 * self.n_kv_heads * self.head_dim, bias=config.bias
            )
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attention_backend = config.attention_backend
        if config.qk_norm:
            if config.attention_backend == "mfa-varlen":
                raise ValueError("qk_norm is not supported with the mfa-varlen attention backend")
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.addmm_residual_projections = config.addmm_residual_projections
        self.contiguous_linear_inputs = config.contiguous_linear_inputs
        self.compact_valid_projections = config.compact_valid_projections

    def __call__(
        self,
        x: mx.array,
        residual: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
        varlen_indices: mx.array | None = None,
        varlen_cu_seqlens: mx.array | None = None,
        valid_indices: mx.array | None = None,
        valid_mask: mx.array | None = None,
        *,
        return_cache: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, T, C = x.shape
        used_varlen_qkv = False
        linear_x = mx.contiguous(x) if self.contiguous_linear_inputs else x
        use_compact_proj = (
            self.compact_valid_projections
            and valid_indices is not None
            and valid_mask is not None
            and self.attn_dropout_p == 0.0
            and cache is None
            and not return_cache
        )
        if self.qkv is not None:
            if use_compact_proj:
                qkv = _scatter_compact_rows(
                    self.qkv(_compact_rows(linear_x, valid_indices)),
                    valid_indices,
                    valid_mask,
                    shape=(B, T, 3 * C),
                )
            else:
                qkv = self.qkv(linear_x)
            if (
                self.attention_backend == "mfa-varlen"
                and cache is None
                and varlen_indices is not None
                and varlen_cu_seqlens is not None
                and (not self.training or self.attn_dropout_p == 0.0)
            ):
                qkv_packed = _pack_varlen_flat(qkv, varlen_indices)[None, :, :]
                y_packed = _mfa_flash_attention_varlen_qkv_packed()(
                    qkv_packed,
                    varlen_cu_seqlens,
                    varlen_cu_seqlens,
                    T,
                    T,
                    scale=self.scale,
                    causal=True,
                    num_heads=self.n_heads,
                    num_kv_heads=self.n_kv_heads,
                )
                y = _unpack_varlen_heads(
                    y_packed,
                    varlen_indices,
                    rows=B * T,
                    heads=self.n_heads,
                    head_dim=self.head_dim,
                ).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
                used_varlen_qkv = True
            else:
                q, k, v = mx.split(qkv, 3, axis=-1)
        else:
            if use_compact_proj:
                q = _scatter_compact_rows(
                    self.q_proj(_compact_rows(linear_x, valid_indices)),
                    valid_indices,
                    valid_mask,
                    shape=(B, T, C),
                )
                kv = _scatter_compact_rows(
                    self.kv_proj(_compact_rows(linear_x, valid_indices)),
                    valid_indices,
                    valid_mask,
                    shape=(B, T, 2 * self.n_kv_heads * self.head_dim),
                )
            else:
                q = self.q_proj(linear_x)
                kv = self.kv_proj(linear_x)
            k, v = mx.split(kv, 2, axis=-1)
        if used_varlen_qkv:
            new_cache = None
        else:
            q = q.reshape(B, T, self.n_heads, self.head_dim)
            k = k.reshape(B, T, self.n_kv_heads, self.head_dim)
            # QK-norm normalizes over head_dim (the last axis) before the transpose.
            if self.q_norm is not None:
                q = self.q_norm(q)
                k = self.k_norm(k)
            q = q.transpose(0, 2, 1, 3)
            k = k.transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

            if cache is not None:
                k_prev, v_prev = cache
                k = mx.concatenate([k_prev, k], axis=2)
                v = mx.concatenate([v_prev, v], axis=2)
                new_cache = (k, v) if return_cache else None
                # MLX's causal mask is lower-right aligned, so it handles both
                # the usual one-token decode and cached chunks with T > 1. The
                # latter must not let an early query see later keys in its chunk.
                y = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=self.scale,
                    mask="causal",
                )
            else:
                new_cache = (k, v) if return_cache else None
                if self.training and self.attn_dropout_p > 0.0:
                    y = _attention_with_dropout(
                        q,
                        k,
                        v,
                        scale=self.scale,
                        mask=attention_mask,
                        dropout=self.attn_dropout,
                    )
                elif self.attention_backend == "mfa":
                    y = _mfa_flash_attention()(
                        q,
                        k,
                        v,
                        scale=self.scale,
                        causal=attention_mask is None,
                        attn_bias=attention_mask,
                        backend="auto",
                    )
                else:
                    y = mx.fast.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        scale=self.scale,
                        mask=attention_mask if attention_mask is not None else "causal",
                    )

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        if self.contiguous_linear_inputs:
            y = mx.contiguous(y)
        if use_compact_proj:
            out = _scatter_compact_rows(
                self.out_proj(_compact_rows(y, valid_indices)),
                valid_indices,
                valid_mask,
                shape=(B, T, C),
            )
            return out, new_cache
        if (
            residual is not None
            and self.addmm_residual_projections
            and self.attn_dropout_p == 0.0
            and cache is None
            and not return_cache
        ):
            return _linear_addmm_residual(residual, y, self.out_proj), new_cache
        return self.resid_dropout(self.out_proj(y)), new_cache


class MLPMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.mlp_type = config.mlp_type
        self.gelu_variant = config.gelu_variant
        if self.mlp_type == "swiglu":
            hidden = config.swiglu_hidden or config.d_ff
            self.swiglu_hidden = hidden
            self.gate_up = nn.Linear(config.d_model, 2 * hidden, bias=config.bias)
            self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
            self.fc1 = None
            self.fc2 = None
        else:
            self.swiglu_hidden = 0
            self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
            self.gate_up = None
            self.down = None
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
        self.addmm_residual_projections = config.addmm_residual_projections
        self.mlp_addmm_linears = getattr(config, "mlp_addmm_linears", False)
        self.contiguous_linear_inputs = config.contiguous_linear_inputs

    def __call__(self, x: mx.array, residual: mx.array | None = None) -> mx.array:
        x = mx.contiguous(x) if self.contiguous_linear_inputs else x
        if self.mlp_type == "swiglu":
            gate_up = _linear_addmm_zero(x, self.gate_up) if self.mlp_addmm_linears else self.gate_up(x)
            gate, up = mx.split(gate_up, 2, axis=-1)
            h = nn.silu(gate) * up
            if self.contiguous_linear_inputs:
                h = mx.contiguous(h)
            if (
                residual is not None
                and self.addmm_residual_projections
                and self.dropout_p == 0.0
            ):
                return _linear_addmm_residual(residual, h, self.down)
            out = _linear_addmm_zero(h, self.down) if self.mlp_addmm_linears else self.down(h)
            return self.dropout(out)
        h = _linear_addmm_zero(x, self.fc1) if self.mlp_addmm_linears else self.fc1(x)
        h = _gelu_fast(h) if self.gelu_variant == "fast" else _gelu_exact(h)
        if self.contiguous_linear_inputs:
            h = mx.contiguous(h)
        if (
            residual is not None
            and self.addmm_residual_projections
            and self.dropout_p == 0.0
        ):
            return _linear_addmm_residual(residual, h, self.fc2)
        out = _linear_addmm_zero(h, self.fc2) if self.mlp_addmm_linears else self.fc2(h)
        return self.dropout(out)


def _linear_addmm_residual(residual: mx.array, x: mx.array, linear: nn.Linear) -> mx.array:
    residual_2d = residual.reshape(-1, residual.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    c = residual_2d
    if "bias" in linear:
        c = c + linear["bias"]
    return mx.addmm(c, x_2d, linear["weight"].T).reshape(residual.shape)


def _linear_addmm_zero(x: mx.array, linear: nn.Linear) -> mx.array:
    x_2d = x.reshape(-1, x.shape[-1])
    c = mx.zeros((), dtype=x.dtype)
    if "bias" in linear:
        c = linear["bias"]
    out = mx.addmm(c, x_2d, linear["weight"].T, beta=0.0 if "bias" not in linear else 1.0)
    return out.reshape(*x.shape[:-1], linear["weight"].shape[0])


_RESIDUAL_RMSNORM_KERNELS = {}


def _residual_rmsnorm_kernel(d_model: int, eps: float = 1e-5, threads: int = 256):
    key = (int(d_model), float(eps), int(threads))
    kernel = _RESIDUAL_RMSNORM_KERNELS.get(key)
    if kernel is not None:
        return kernel
    source = f"""
        uint global = thread_position_in_grid.x;
        uint row = global / {threads};
        uint tid = thread_position_in_threadgroup.x;
        constexpr uint D = {int(d_model)};
        constexpr uint THREADS = {int(threads)};
        threadgroup float partial[THREADS];

        float sumsq = 0.0f;
        uint base = row * D;
        for (uint col = tid; col < D; col += THREADS) {{
            float v = float(x[base + col]) + float(update[base + col]);
            residual[base + col] = T(v);
            sumsq += v * v;
        }}
        partial[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                partial[tid] += partial[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv_rms = metal::rsqrt(partial[0] / float(D) + {eps:.12g}f);
        for (uint col = tid; col < D; col += THREADS) {{
            float v = float(residual[base + col]);
            normed[base + col] = T(v * inv_rms * float(weight[col]));
        }}
    """
    kernel = mx.fast.metal_kernel(
        name=f"residual_add_rmsnorm_d{int(d_model)}_t{int(threads)}",
        input_names=["x", "update", "weight"],
        output_names=["residual", "normed"],
        source=source,
        ensure_row_contiguous=True,
    )
    _RESIDUAL_RMSNORM_KERNELS[key] = kernel
    return kernel


@mx.custom_function
def _residual_add_rms_norm(
    x: mx.array,
    update: mx.array,
    weight: mx.array,
) -> tuple[mx.array, mx.array]:
    threads = 256
    kernel = _residual_rmsnorm_kernel(x.shape[-1], eps=1e-5, threads=threads)
    residual, normed = kernel(
        inputs=[x, update, weight],
        template=[("T", x.dtype)],
        grid=(x.shape[0] * x.shape[1] * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return residual, normed


@_residual_add_rms_norm.vjp
def _residual_add_rms_norm_vjp(primals, cotangents, output):
    x, update, weight = primals
    cot_residual, cot_normed = cotangents
    z = (x + update).astype(mx.float32)
    dy = cot_normed.astype(mx.float32)
    w = weight.astype(mx.float32)
    inv_rms = mx.rsqrt(mx.mean(z * z, axis=-1, keepdims=True) + 1e-5)
    weighted_dy = dy * w
    row_dot = mx.sum(weighted_dy * z, axis=-1, keepdims=True)
    grad_norm_input = weighted_dy * inv_rms - z * (inv_rms * inv_rms * inv_rms) * row_dot / z.shape[-1]
    grad_direct = cot_residual.astype(mx.float32)
    grad_z = grad_direct + grad_norm_input
    reduce_axes = tuple(range(z.ndim - 1))
    grad_weight = mx.sum(dy * z * inv_rms, axis=reduce_axes)
    return (
        grad_z.astype(x.dtype),
        grad_z.astype(update.dtype),
        grad_weight.astype(weight.dtype),
    )


def _compact_rows(x: mx.array, indices: mx.array) -> mx.array:
    return mx.take(x.reshape(-1, x.shape[-1]), indices, axis=0)


def _scatter_compact_rows(
    values: mx.array,
    indices: mx.array,
    mask: mx.array,
    *,
    shape: tuple[int, int, int],
) -> mx.array:
    masked_values = values * mask.astype(values.dtype)[:, None]
    out = mx.zeros((shape[0] * shape[1], shape[2]), dtype=values.dtype)
    out = mx.put_along_axis(out, indices[:, None], masked_values, axis=0)
    return out.reshape(shape)


class TransformerBlockMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        norm = nn.RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm(config.d_model) if config.norm_type == "rmsnorm" else norm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttentionMLX(config)
        self.ln2 = norm(config.d_model) if config.norm_type == "rmsnorm" else norm(config.d_model, bias=config.bias)
        self.mlp = MLPMLX(config)
        self.residual_type = config.residual_type
        self.addmm_residual_projections = config.addmm_residual_projections
        self.compact_valid_projections = config.compact_valid_projections
        self.mlp_checkpointing = config.mlp_checkpointing
        self.fused_residual_rmsnorm = (
            config.fused_residual_rmsnorm
            and config.norm_type == "rmsnorm"
            and config.residual_type == "serial"
        )
        self._checkpoint_mlp = nn_utils.checkpoint(self.mlp) if config.mlp_checkpointing else None

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
        varlen_indices: mx.array | None = None,
        varlen_cu_seqlens: mx.array | None = None,
        valid_indices: mx.array | None = None,
        valid_mask: mx.array | None = None,
        *,
        return_cache: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        h = self.ln1(x)
        fuse_attn_residual = (
            self.addmm_residual_projections
            and self.residual_type == "serial"
            and cache is None
            and not return_cache
        )
        attn_out, new_cache = self.attn(
            h,
            residual=x if fuse_attn_residual else None,
            cache=cache,
            attention_mask=attention_mask,
            varlen_indices=varlen_indices,
            varlen_cu_seqlens=varlen_cu_seqlens,
            valid_indices=valid_indices if self.compact_valid_projections else None,
            valid_mask=valid_mask if self.compact_valid_projections else None,
            return_cache=return_cache,
        )
        mlp = self._checkpoint_mlp if self.mlp_checkpointing and cache is None and not return_cache else self.mlp
        use_compact_mlp = (
            valid_indices is not None
            and valid_mask is not None
            and cache is None
            and not return_cache
            and not self.mlp_checkpointing
        )
        # Parallel residual topology is part of the model architecture, not a
        # training-only optimization. Cache warmup and incremental decode must
        # use the same x + attention(norm(x)) + mlp(norm(x)) equation as a full
        # forward pass; falling through here silently changed cached inference
        # into a serial-residual model.
        if self.residual_type == "parallel":
            if use_compact_mlp:
                mlp_out = _scatter_compact_rows(
                    mlp(_compact_rows(h, valid_indices)),
                    valid_indices,
                    valid_mask,
                    shape=x.shape,
                )
            else:
                mlp_out = mlp(h)
            x = x + attn_out + mlp_out
            return x, new_cache
        if (
            self.fused_residual_rmsnorm
            and not fuse_attn_residual
            and cache is None
            and not return_cache
        ):
            x, h2 = _residual_add_rms_norm(x, attn_out, self.ln2["weight"])
        else:
            x = attn_out if fuse_attn_residual else x + attn_out
            h2 = self.ln2(x)
        if use_compact_mlp:
            mlp_out = _scatter_compact_rows(
                mlp(_compact_rows(h2, valid_indices)),
                valid_indices,
                valid_mask,
                shape=x.shape,
            )
        else:
            if self.addmm_residual_projections and cache is None and not return_cache:
                x = mlp(h2, residual=x)
                return x, new_cache
            mlp_out = mlp(h2)
        x = x + mlp_out
        return x, new_cache


class SpakieGPTMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = [TransformerBlockMLX(config) for _ in range(config.n_layers)]
        # Wrap blocks for activation checkpointing only when the preset asks
        # for it. Building the wrappers eagerly is cheap, but keeping them only
        # when used avoids surprising aliasing in module-state traversals.
        self._checkpoint_blocks = (
            [nn_utils.checkpoint(block) for block in self.blocks]
            if config.activation_checkpointing
            else None
        )
        self.ln_f = (
            nn.RMSNorm(config.d_model)
            if config.norm_type == "rmsnorm"
            else nn.LayerNorm(config.d_model, bias=config.bias)
        )

        # lm_head shares weights with tok_emb — we don't allocate a separate matrix;
        # forward uses tok_emb.weight directly as the projection.

        self._init_weights(config)

    def _init_weights(self, config: SpakieConfig):
        # Match PyTorch init: Normal(0, 0.02), with residual projections scaled by
        # 0.02 / sqrt(2 * n_layers).
        def _normal(shape, std):
            return mx.random.normal(shape=shape) * std

        # Collect per-submodule overrides in one pass for clarity.
        overrides = {}
        overrides["tok_emb.weight"] = _normal(self.tok_emb.weight.shape, 0.02)
        overrides["pos_emb.weight"] = _normal(self.pos_emb.weight.shape, 0.02)
        overrides["ln_f.weight"] = mx.ones_like(self.ln_f.weight)
        if getattr(self.ln_f, "bias", None) is not None:
            overrides["ln_f.bias"] = mx.zeros_like(self.ln_f.bias)

        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}"
            overrides[f"{prefix}.ln1.weight"] = mx.ones_like(block.ln1.weight)
            if getattr(block.ln1, "bias", None) is not None:
                overrides[f"{prefix}.ln1.bias"] = mx.zeros_like(block.ln1.bias)
            overrides[f"{prefix}.ln2.weight"] = mx.ones_like(block.ln2.weight)
            if getattr(block.ln2, "bias", None) is not None:
                overrides[f"{prefix}.ln2.bias"] = mx.zeros_like(block.ln2.bias)

            if block.attn.qkv is not None:
                overrides[f"{prefix}.attn.qkv.weight"] = _normal(block.attn.qkv.weight.shape, 0.02)
                if config.bias:
                    overrides[f"{prefix}.attn.qkv.bias"] = mx.zeros_like(block.attn.qkv.bias)
            else:
                overrides[f"{prefix}.attn.q_proj.weight"] = _normal(
                    block.attn.q_proj.weight.shape, 0.02
                )
                overrides[f"{prefix}.attn.kv_proj.weight"] = _normal(
                    block.attn.kv_proj.weight.shape, 0.02
                )
                if config.bias:
                    overrides[f"{prefix}.attn.q_proj.bias"] = mx.zeros_like(block.attn.q_proj.bias)
                    overrides[f"{prefix}.attn.kv_proj.bias"] = mx.zeros_like(
                        block.attn.kv_proj.bias
                    )
            overrides[f"{prefix}.attn.out_proj.weight"] = _normal(
                block.attn.out_proj.weight.shape, residual_std
            )
            if config.bias:
                overrides[f"{prefix}.attn.out_proj.bias"] = mx.zeros_like(block.attn.out_proj.bias)
            if block.attn.q_norm is not None:
                overrides[f"{prefix}.attn.q_norm.weight"] = mx.ones_like(block.attn.q_norm.weight)
                overrides[f"{prefix}.attn.k_norm.weight"] = mx.ones_like(block.attn.k_norm.weight)

            if block.mlp.mlp_type == "swiglu":
                overrides[f"{prefix}.mlp.gate_up.weight"] = _normal(
                    block.mlp.gate_up.weight.shape, 0.02
                )
                if config.bias:
                    overrides[f"{prefix}.mlp.gate_up.bias"] = mx.zeros_like(
                        block.mlp.gate_up.bias
                    )
                overrides[f"{prefix}.mlp.down.weight"] = _normal(
                    block.mlp.down.weight.shape, residual_std
                )
                if config.bias:
                    overrides[f"{prefix}.mlp.down.bias"] = mx.zeros_like(block.mlp.down.bias)
            else:
                overrides[f"{prefix}.mlp.fc1.weight"] = _normal(block.mlp.fc1.weight.shape, 0.02)
                if config.bias:
                    overrides[f"{prefix}.mlp.fc1.bias"] = mx.zeros_like(block.mlp.fc1.bias)
                overrides[f"{prefix}.mlp.fc2.weight"] = _normal(
                    block.mlp.fc2.weight.shape, residual_std
                )
                if config.bias:
                    overrides[f"{prefix}.mlp.fc2.bias"] = mx.zeros_like(block.mlp.fc2.bias)

        from mlx.utils import tree_unflatten

        self.update(tree_unflatten(list(overrides.items())))

    def __call__(
        self,
        idx: mx.array,
        targets: mx.array | None = None,
        cache: list[tuple[mx.array, mx.array]] | None = None,
        *,
        cache_offset: int = 0,
        return_cache: bool = False,
        ignore_index: int | None = -100,
        allow_fused_ce: bool = True,
        segment_ids: mx.array | None = None,
        position_ids: mx.array | None = None,
        loss_indices: mx.array | None = None,
        loss_targets: mx.array | None = None,
        loss_mask: mx.array | None = None,
        varlen_indices: mx.array | None = None,
        varlen_cu_seqlens: mx.array | None = None,
        valid_indices: mx.array | None = None,
        valid_mask: mx.array | None = None,
    ):
        B, T = idx.shape
        assert T + cache_offset <= self.config.max_seq_len, (
            f"Sequence length {T + cache_offset} exceeds max {self.config.max_seq_len}"
        )

        pos = position_ids if position_ids is not None else mx.arange(cache_offset, cache_offset + T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        attention_mask = (
            _block_causal_attention_mask(segment_ids, dtype=x.dtype)
            if segment_ids is not None
            else None
        )

        use_cache = cache is not None
        new_caches: list[tuple[mx.array, mx.array] | None] | None = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if use_cache else None
            if self.training and self.config.activation_checkpointing and not use_cache and not return_cache:
                # Activation checkpointing path — only used during training (no KV cache).
                x = self._checkpoint_blocks[i](
                    x,
                    None,
                    attention_mask,
                    varlen_indices,
                    varlen_cu_seqlens,
                    valid_indices,
                    valid_mask,
                    return_cache=False,
                )[0]
            else:
                x, nc = block(
                    x,
                    cache=block_cache,
                    attention_mask=attention_mask,
                    varlen_indices=varlen_indices,
                    varlen_cu_seqlens=varlen_cu_seqlens,
                    valid_indices=valid_indices,
                    valid_mask=valid_mask,
                    return_cache=return_cache,
                )
                if return_cache and new_caches is not None:
                    new_caches.append(nc)

        x = self.ln_f(x)
        W = self.tok_emb.weight  # tied lm_head: (vocab_size, d_model)

        # When `targets` is provided in training, we don't need the full
        # (B, T, vocab_size) logits tensor — only the per-token loss. At
        # vocab_size=16384 and B*T~50k that tensor is ~1.6 GB in bf16,
        # which is the dominant chunk of memory traffic for fwd+bwd. Compute
        # the matmul + cross-entropy in chunks along (B*T) so the peak
        # logits tensor stays small; this is purely a compute/memory
        # optimization (not a numerical change) and is skipped whenever the
        # caller actually wants logits back (inference, no targets).
        if targets is None or return_cache or cache is not None:
            logits = x @ W.T
            loss = None
            if targets is not None:
                flat_logits = logits.flatten(0, 1)
                flat_targets = targets.flatten()
                loss = _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, logits.dtype
                )
            return logits, loss, new_caches

        return None, self._loss_from_hidden(
            x,
            targets,
            ignore_index=ignore_index,
            allow_fused_ce=allow_fused_ce,
            loss_indices=loss_indices,
            loss_targets=loss_targets,
            loss_mask=loss_mask,
        ), new_caches

    def _loss_from_hidden(
        self,
        x: mx.array,
        targets: mx.array,
        *,
        ignore_index: int | None,
        allow_fused_ce: bool = True,
        loss_indices: mx.array | None = None,
        loss_targets: mx.array | None = None,
        loss_mask: mx.array | None = None,
    ) -> mx.array:
        W = self.tok_emb.weight
        flat_x = x.flatten(0, 1)
        flat_targets = targets.flatten()
        if loss_indices is not None:
            if loss_targets is None or loss_mask is None:
                raise ValueError("loss_indices requires loss_targets and loss_mask")
            loss_x = mx.take(flat_x, loss_indices, axis=0)
            return _indexed_cross_entropy_mean(loss_x, W, loss_targets, loss_mask)

        N = flat_x.shape[0]
        chunk = self.config.loss_chunk_size or N
        if chunk <= 0 or chunk >= N:
            if self.config.loss_layout == "custom" and ignore_index is None:
                # The fused CE launches a Metal CustomKernel, which MLX cannot
                # vmap (`[Primitive::vmap] Not implemented for CustomKernel`).
                # Callers that run this loss under `mx.vmap` (e.g. the vmap
                # gradient-accumulation path) pass allow_fused_ce=False to fall
                # back to the pure-MLX custom path, which is vmap-safe and
                # numerically identical.
                if allow_fused_ce and getattr(self.config, "fused_cross_entropy", False):
                    from model.fused_ce_mlx import fused_linear_cross_entropy_mean
                    return fused_linear_cross_entropy_mean(flat_x, W, flat_targets)
                return _linear_cross_entropy_mean(flat_x, W, flat_targets)
            if self.config.loss_layout == "flat":
                flat_logits = flat_x @ W.T
                return _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, flat_logits.dtype
                )
            logits = x @ W.T
            if ignore_index is None:
                return nn.losses.cross_entropy(logits, targets, axis=-1, reduction="mean")
            else:
                flat_logits = logits.reshape(-1, logits.shape[-1])
                return _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, flat_logits.dtype
                )

        loss_sum = mx.zeros((), dtype=mx.float32)
        valid_count = mx.zeros((), dtype=mx.float32)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            cx = flat_x[i:j]
            ct = flat_targets[i:j]
            clogits = cx @ W.T
            if ignore_index is None:
                cl = nn.losses.cross_entropy(clogits, ct, reduction="sum").astype(mx.float32)
                loss_sum = loss_sum + cl
                valid_count = valid_count + mx.array(float(j - i), dtype=mx.float32)
            else:
                cmask = (ct != ignore_index).astype(clogits.dtype)
                csafe = mx.where(ct != ignore_index, ct, mx.zeros_like(ct))
                cper = nn.losses.cross_entropy(clogits, csafe, reduction="none")
                loss_sum = loss_sum + (cper * cmask).sum().astype(mx.float32)
                valid_count = valid_count + cmask.sum().astype(mx.float32)
        denom = mx.maximum(valid_count, mx.array(1.0, dtype=mx.float32))
        return loss_sum / denom


def _ce_with_optional_mask(
    flat_logits: mx.array,
    flat_targets: mx.array,
    ignore_index: int | None,
    logits_dtype,
) -> mx.array:
    if ignore_index is None:
        return nn.losses.cross_entropy(flat_logits, flat_targets, reduction="mean")
    mask = (flat_targets != ignore_index).astype(logits_dtype)
    safe = mx.where(flat_targets != ignore_index, flat_targets, mx.zeros_like(flat_targets))
    per_tok = nn.losses.cross_entropy(flat_logits, safe, reduction="none")
    denom = mx.maximum(mask.sum(), mx.array(1.0, dtype=mask.dtype))
    return (per_tok * mask).sum() / denom


def _indexed_cross_entropy_mean(
    hidden: mx.array,
    weight: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> mx.array:
    logits = hidden @ weight.T
    per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")
    loss_mask = mask.astype(per_tok.dtype)
    denom = mx.maximum(loss_mask.sum(), mx.array(1.0, dtype=loss_mask.dtype))
    return (per_tok * loss_mask).sum() / denom


def _pack_varlen_heads(x: mx.array, indices: mx.array) -> mx.array:
    B, H, T, D = x.shape
    flat = x.transpose(0, 2, 1, 3).reshape(B * T, H, D)
    packed = mx.take(flat, indices, axis=0)
    return packed.transpose(1, 0, 2)[None, :, :, :]


def _pack_varlen_flat(x: mx.array, indices: mx.array) -> mx.array:
    B, T, C = x.shape
    flat = x.reshape(B * T, C)
    return mx.take(flat, indices, axis=0)


def _unpack_varlen_heads(
    packed: mx.array,
    indices: mx.array,
    *,
    rows: int,
    heads: int,
    head_dim: int,
) -> mx.array:
    values = packed[0].transpose(1, 0, 2)
    out = mx.zeros((rows, heads, head_dim), dtype=values.dtype)
    scatter_idx = indices[:, None, None]
    return mx.put_along_axis(out, scatter_idx, values, axis=0)


def _block_causal_attention_mask(segment_ids: mx.array, *, dtype) -> mx.array:
    B, T = segment_ids.shape
    q_pos = mx.arange(T)[:, None]
    k_pos = mx.arange(T)[None, :]
    causal = k_pos <= q_pos
    q_seg = segment_ids[:, :, None]
    k_seg = segment_ids[:, None, :]
    valid = (q_seg >= 0) & (q_seg == k_seg) & causal
    return valid[:, None, :, :]
