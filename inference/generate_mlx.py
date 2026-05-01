"""Autoregressive generation with KV cache + top-k/top-p sampling (MLX)."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.transformer_mlx import SpakieGPTMLX
from tokenizer.train_tokenizer import SpakieTokenizer


def _apply_top_k_top_p_np(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    """Filter a [V] numpy logit vector with temperature, top-k, and top-p."""
    logits = logits / temperature
    vocab = logits.shape[-1]

    if top_k and top_k > 0:
        k = min(top_k, vocab)
        threshold = np.partition(logits, vocab - k)[vocab - k]
        logits = np.where(logits < threshold, -np.inf, logits)

    if top_p < 1.0:
        order = np.argsort(-logits)
        sorted_logits = logits[order]
        # softmax
        m = sorted_logits.max()
        probs = np.exp(sorted_logits - m)
        probs = probs / probs.sum()
        cumulative = np.cumsum(probs)
        keep = (cumulative - probs) < top_p
        sorted_logits = np.where(keep, sorted_logits, -np.inf)
        # scatter back
        out = np.empty_like(logits)
        out[order] = sorted_logits
        logits = out

    return logits


def _sample_host(logits_vec: np.ndarray) -> int:
    """Sample a token id from a host-side [V] logits vector."""
    m = logits_vec.max()
    if not np.isfinite(m):
        return int(np.argmax(logits_vec))
    probs = np.exp(logits_vec - m)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(
    model: SpakieGPTMLX,
    tokenizer: SpakieTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> list[int]:
    model.eval()
    generated: list[int] = []
    token_counts: Counter[int] = Counter()
    stop_tokens = {
        tokenizer.eos_id,
        tokenizer.user_id,
        tokenizer.assistant_id,
        tokenizer.system_id,
    }
    banned = (
        tokenizer.user_id,
        tokenizer.assistant_id,
        tokenizer.system_id,
        tokenizer.json_id,
        tokenizer.pad_id,
    )

    # Warm the KV cache with the prompt (truncated to max_seq_len).
    truncated = prompt_ids[-model.config.max_seq_len :]
    idx = mx.array([truncated], dtype=mx.int32)
    logits, _, cache = model(idx, cache=None, cache_offset=0, return_cache=True)
    mx.eval(logits)
    cache_offset = idx.shape[1]

    for _ in range(max_new_tokens):
        # Pull just the last-step logits to host.
        last = np.asarray(logits[0, -1, :].astype(mx.float32))

        if repetition_penalty != 1.0:
            for tok, count in token_counts.items():
                penalty = repetition_penalty ** count
                v = last[tok]
                last[tok] = v / penalty if v > 0 else v * penalty

        for tok in banned:
            last[tok] = -np.inf

        filtered = _apply_top_k_top_p_np(last, temperature, top_k, top_p)
        token = _sample_host(filtered)
        if token in stop_tokens:
            break
        generated.append(token)
        token_counts[token] += 1

        if cache_offset + 1 > model.config.max_seq_len:
            break

        idx = mx.array([[token]], dtype=mx.int32)
        logits, _, cache = model(idx, cache=cache, cache_offset=cache_offset, return_cache=True)
        cache_offset += 1

    return generated


def generate_json(
    model: SpakieGPTMLX,
    tokenizer: SpakieTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 512,
    temperature: float = 0.5,
    top_k: int = 50,
    top_p: float = 0.9,
    max_retries: int = 3,
) -> str | None:
    json_prompt = [tokenizer.json_id] + prompt_ids
    for attempt in range(max_retries):
        ids = generate(
            model,
            tokenizer,
            json_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
        )
        text = tokenizer.decode(ids).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                temperature *= 0.8
            continue
    return None
