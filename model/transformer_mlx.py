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


def _gelu_exact(x: mx.array) -> mx.array:
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2


def _gelu_fast(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


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
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
        *,
        return_cache: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, T, C = x.shape
        if self.qkv is not None:
            qkv = self.qkv(x)
            q, k, v = mx.split(qkv, 3, axis=-1)
        else:
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            k, v = mx.split(kv, 2, axis=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k_prev, v_prev = cache
            k = mx.concatenate([k_prev, k], axis=2)
            v = mx.concatenate([v_prev, v], axis=2)
            new_cache = (k, v) if return_cache else None
            # When using cache the incremental q has length T (usually 1). Attending
            # to the full cached k/v is inherently causal for decoding, so no mask.
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            new_cache = (k, v) if return_cache else None
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            )

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
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

    def __call__(self, x: mx.array) -> mx.array:
        if self.mlp_type == "swiglu":
            gate_up = self.gate_up(x)
            gate, up = mx.split(gate_up, 2, axis=-1)
            return self.dropout(self.down(nn.silu(gate) * up))
        h = self.fc1(x)
        h = _gelu_fast(h) if self.gelu_variant == "fast" else _gelu_exact(h)
        return self.dropout(self.fc2(h))


class TransformerBlockMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        norm = nn.RMSNorm if config.norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm(config.d_model) if config.norm_type == "rmsnorm" else norm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttentionMLX(config)
        self.ln2 = norm(config.d_model) if config.norm_type == "rmsnorm" else norm(config.d_model, bias=config.bias)
        self.mlp = MLPMLX(config)
        self.residual_type = config.residual_type

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
        *,
        return_cache: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        h = self.ln1(x)
        attn_out, new_cache = self.attn(h, cache=cache, return_cache=return_cache)
        if self.residual_type == "parallel" and cache is None and not return_cache:
            x = x + attn_out + self.mlp(h)
            return x, new_cache
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
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
    ):
        B, T = idx.shape
        assert T + cache_offset <= self.config.max_seq_len, (
            f"Sequence length {T + cache_offset} exceeds max {self.config.max_seq_len}"
        )

        pos = mx.arange(cache_offset, cache_offset + T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        use_cache = cache is not None
        new_caches: list[tuple[mx.array, mx.array] | None] | None = [] if return_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if use_cache else None
            if self.training and self.config.activation_checkpointing and not use_cache and not return_cache:
                # Activation checkpointing path — only used during training (no KV cache).
                x = self._checkpoint_blocks[i](x, None, return_cache=False)[0]
            else:
                x, nc = block(x, cache=block_cache, return_cache=return_cache)
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
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_targets = targets.reshape(-1)
                loss = _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, logits.dtype
                )
            return logits, loss, new_caches

        flat_x = x.reshape(-1, x.shape[-1])
        flat_targets = targets.reshape(-1)
        N = flat_x.shape[0]
        chunk = self.config.loss_chunk_size or N
        if chunk <= 0 or chunk >= N:
            if self.config.loss_layout == "custom" and ignore_index is None:
                return None, _linear_cross_entropy_mean(flat_x, W, flat_targets), new_caches
            if self.config.loss_layout == "flat":
                flat_logits = flat_x @ W.T
                loss = _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, flat_logits.dtype
                )
                return None, loss, new_caches
            logits = x @ W.T
            if ignore_index is None:
                loss = nn.losses.cross_entropy(logits, targets, axis=-1, reduction="mean")
            else:
                flat_logits = logits.reshape(-1, logits.shape[-1])
                loss = _ce_with_optional_mask(
                    flat_logits, flat_targets, ignore_index, flat_logits.dtype
                )
            return None, loss, new_caches

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
        return None, loss_sum / denom, new_caches


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
