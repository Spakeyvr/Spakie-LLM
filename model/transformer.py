"""GPT-style transformer: CausalSelfAttention, MLP, TransformerBlock, SpakieGPT."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


_SDPA_SUPPORTS_GQA: bool | None = None


def apply_rotary_emb(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
    inv_freq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Llama-style half-rotation RoPE to ``[B, T, H, D]`` Q/K tensors."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")
    half_dim = head_dim // 2
    positions = position_ids
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if inv_freq is None:
        inv_freq = torch.exp(
            -math.log(theta)
            * torch.arange(half_dim, dtype=torch.float32, device=x.device)
            / half_dim
        )
    angles = positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = angles.cos().to(dtype=x.dtype).unsqueeze(2)
    sin = angles.sin().to(dtype=x.dtype).unsqueeze(2)
    first, second = x[..., :half_dim], x[..., half_dim:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        half_dim = self.head_dim // 2
        rope_inv_freq = torch.exp(
            -math.log(self.rope_theta)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("rope_inv_freq", rope_inv_freq, persistent=False)

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
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_cache: bool = False,
    ):
        B, T, C = x.shape
        if self.qkv is not None:
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
        else:
            q = self.q_proj(x)
            kv = self.kv_proj(x)
            k, v = kv.split(self.n_kv_heads * self.head_dim, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        # QK-norm normalizes over head_dim (the last axis) before the transpose.
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if position_ids is None:
            position_ids = torch.arange(T, dtype=torch.long, device=x.device)
        q = apply_rotary_emb(q, position_ids, self.rope_theta, self.rope_inv_freq)
        k = apply_rotary_emb(k, position_ids, self.rope_theta, self.rope_inv_freq)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            cached_k, cached_v = cache
            k = torch.cat((cached_k, k), dim=2)
            v = torch.cat((cached_v, v), dim=2)
        new_cache = (k, v) if return_cache else None

        if cache is None:
            attention_mask = None
            is_causal = True
        else:
            # SDPA's rectangular causal alignment is backend/version sensitive.
            # Build the lower-right mask explicitly for cached chunks; the usual
            # one-token decode consequently attends to every cached key.
            key_length = k.shape[2]
            past_length = key_length - T
            query_positions = torch.arange(T, device=x.device)[:, None] + past_length
            key_positions = torch.arange(key_length, device=x.device)[None, :]
            attention_mask = key_positions <= query_positions
            is_causal = False

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if self.n_kv_heads == self.n_heads:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask,
                is_causal=is_causal, dropout_p=dropout_p,
            )
        else:
            global _SDPA_SUPPORTS_GQA
            if _SDPA_SUPPORTS_GQA is not False:
                try:
                    y = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attention_mask,
                        is_causal=is_causal, dropout_p=dropout_p, enable_gqa=True,
                    )
                    _SDPA_SUPPORTS_GQA = True
                except TypeError as exc:
                    if "enable_gqa" not in str(exc):
                        raise
                    _SDPA_SUPPORTS_GQA = False
            if _SDPA_SUPPORTS_GQA is False:
                repeat = self.n_heads // self.n_kv_heads
                expanded_k = k.repeat_interleave(repeat, dim=1)
                expanded_v = v.repeat_interleave(repeat, dim=1)
                y = F.scaled_dot_product_attention(
                    q, expanded_k, expanded_v, attn_mask=attention_mask,
                    is_causal=is_causal, dropout_p=dropout_p,
                )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.resid_dropout(self.out_proj(y))
        return (output, new_cache) if return_cache else output


class MLP(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        hidden = config.swiglu_hidden or config.d_ff
        self.gate_up = nn.Linear(config.d_model, 2 * hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.ln1 = (
            nn.RMSNorm(config.d_model, eps=1e-5)
            if config.norm_type == "rmsnorm"
            else nn.LayerNorm(config.d_model, bias=config.bias)
        )
        self.attn = CausalSelfAttention(config)
        self.ln2 = (
            nn.RMSNorm(config.d_model, eps=1e-5)
            if config.norm_type == "rmsnorm"
            else nn.LayerNorm(config.d_model, bias=config.bias)
        )
        self.mlp = MLP(config)
        self.residual_type = config.residual_type

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_cache: bool = False,
    ):
        h = self.ln1(x)
        attention_result = self.attn(
            h, position_ids, cache=cache, return_cache=return_cache
        )
        if return_cache:
            attn_out, new_cache = attention_result
        else:
            attn_out = attention_result
            new_cache = None
        if self.residual_type == "parallel":
            output = x + attn_out + self.mlp(h)
            return (output, new_cache) if return_cache else output
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return (x, new_cache) if return_cache else x


class SpakieGPT(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = (
            nn.RMSNorm(config.d_model, eps=1e-5)
            if config.norm_type == "rmsnorm"
            else nn.LayerNorm(config.d_model, bias=config.bias)
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale residual projections
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        cache_offset: int = 0,
        return_cache: bool = False,
    ):
        B, T = idx.shape
        if T + cache_offset > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T + cache_offset} exceeds max {self.config.max_seq_len}"
            )
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("KV cache layer count does not match the model")

        pos = (
            position_ids
            if position_ids is not None
            else torch.arange(cache_offset, cache_offset + T, dtype=torch.long, device=idx.device)
        )
        x = self.tok_emb(idx)
        x = self.drop(x)

        new_caches = [] if return_cache else None
        for block_index, block in enumerate(self.blocks):
            block_cache = cache[block_index] if cache is not None else None
            if (
                self.training
                and self.config.activation_checkpointing
                and cache is None
                and not return_cache
            ):
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                block_result = block(
                    x, pos, cache=block_cache, return_cache=return_cache
                )
                if return_cache:
                    x, new_cache = block_result
                    new_caches.append(new_cache)
                else:
                    x = block_result

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        if return_cache:
            return logits, loss, new_caches
        return logits, loss
