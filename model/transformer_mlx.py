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


class CausalSelfAttentionMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            k_prev, v_prev = cache
            k = mx.concatenate([k_prev, k], axis=2)
            v = mx.concatenate([v_prev, v], axis=2)
            new_cache = (k, v)
            # When using cache the incremental q has length T (usually 1). Attending
            # to the full cached k/v is inherently causal for decoding, so no mask.
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            new_cache = (k, v)
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            )

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(y)), new_cache


class MLPMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.fc2(nn.gelu(self.fc1(x))))


class TransformerBlockMLX(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttentionMLX(config)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLPMLX(config)

    def __call__(
        self,
        x: mx.array,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        attn_out, new_cache = self.attn(self.ln1(x), cache=cache)
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
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)

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

            overrides[f"{prefix}.attn.qkv.weight"] = _normal(block.attn.qkv.weight.shape, 0.02)
            if config.bias:
                overrides[f"{prefix}.attn.qkv.bias"] = mx.zeros_like(block.attn.qkv.bias)
            overrides[f"{prefix}.attn.out_proj.weight"] = _normal(
                block.attn.out_proj.weight.shape, residual_std
            )
            if config.bias:
                overrides[f"{prefix}.attn.out_proj.bias"] = mx.zeros_like(block.attn.out_proj.bias)

            overrides[f"{prefix}.mlp.fc1.weight"] = _normal(block.mlp.fc1.weight.shape, 0.02)
            if config.bias:
                overrides[f"{prefix}.mlp.fc1.bias"] = mx.zeros_like(block.mlp.fc1.bias)
            overrides[f"{prefix}.mlp.fc2.weight"] = _normal(block.mlp.fc2.weight.shape, residual_std)
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
    ):
        B, T = idx.shape
        assert T + cache_offset <= self.config.max_seq_len, (
            f"Sequence length {T + cache_offset} exceeds max {self.config.max_seq_len}"
        )

        pos = mx.arange(cache_offset, cache_offset + T)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        new_caches: list[tuple[mx.array, mx.array] | None] = []
        for i, block in enumerate(self.blocks):
            block_cache = cache[i] if cache is not None else None
            if self.training and self.config.activation_checkpointing and cache is None:
                # Activation checkpointing path — only used during training (no KV cache).
                x = nn_utils.checkpoint(block)(x, None)[0]
                new_caches.append(None)
            else:
                x, nc = block(x, cache=block_cache)
                new_caches.append(nc)

        x = self.ln_f(x)
        # Tied lm_head: project with tok_emb.weight^T.
        logits = x @ self.tok_emb.weight.T

        loss = None
        if targets is not None:
            # Mask out ignore_index (-100). Replace with 0 so gather is safe; weight=0 there.
            mask = (targets != -100).astype(logits.dtype)
            safe_targets = mx.where(targets != -100, targets, mx.zeros_like(targets))
            per_tok = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                safe_targets.reshape(-1),
                reduction="none",
            )
            mask_flat = mask.reshape(-1)
            denom = mx.maximum(mask_flat.sum(), mx.array(1.0, dtype=mask_flat.dtype))
            loss = (per_tok * mask_flat).sum() / denom

        return logits, loss, new_caches
