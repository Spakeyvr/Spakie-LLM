"""GPT-style transformer: CausalSelfAttention, MLP, TransformerBlock, SpakieGPT."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


def apply_rotary_emb(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Apply Llama-style half-rotation RoPE to ``[B, T, H, D]`` Q/K tensors."""
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")
    half_dim = head_dim // 2
    positions = position_ids
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
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
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.head_dim = config.d_model // config.n_heads
        self.position_encoding = config.position_encoding
        self.rope_theta = config.rope_theta

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
    ) -> torch.Tensor:
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
        if self.position_encoding == "rope":
            if position_ids is None:
                position_ids = torch.arange(T, dtype=torch.long, device=x.device)
            q = apply_rotary_emb(q, position_ids, self.rope_theta)
            k = apply_rotary_emb(k, position_ids, self.rope_theta)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if self.n_kv_heads == self.n_heads:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            try:
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=dropout_p, enable_gqa=True
                )
            except TypeError:
                repeat = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.mlp_type = config.mlp_type
        self.gelu_variant = config.gelu_variant
        if self.mlp_type == "swiglu":
            hidden = config.swiglu_hidden or config.d_ff
            self.gate_up = nn.Linear(config.d_model, 2 * hidden, bias=config.bias)
            self.down = nn.Linear(hidden, config.d_model, bias=config.bias)
            self.fc1 = None
            self.fc2 = None
        else:
            self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
            self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
            self.gate_up = None
            self.down = None
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.dropout(self.down(F.silu(gate) * up))
        h = self.fc1(x)
        if self.gelu_variant == "fast":
            h = h * torch.sigmoid(1.702 * h)
        else:
            h = F.gelu(h)
        return self.dropout(self.fc2(h))


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
    ) -> torch.Tensor:
        h = self.ln1(x)
        attn_out = self.attn(h, position_ids)
        if self.residual_type == "parallel":
            return x + attn_out + self.mlp(h)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class SpakieGPT(nn.Module):
    def __init__(self, config: SpakieConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = (
            nn.Embedding(config.max_seq_len, config.d_model)
            if config.position_encoding == "learned"
            else None
        )
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
            if block.mlp.mlp_type == "swiglu":
                nn.init.normal_(block.mlp.down.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
            else:
                nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

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
    ):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        pos = (
            position_ids
            if position_ids is not None
            else torch.arange(0, T, dtype=torch.long, device=idx.device)
        )
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            if self.training and self.config.activation_checkpointing:
                x = checkpoint(block, x, pos, use_reentrant=False)
            else:
                x = block(x, pos)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss
