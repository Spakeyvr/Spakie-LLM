"""Autoregressive generation with KV cache + top-k/top-p sampling (MLX)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.transformer_mlx import SpakieGPTMLX
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.generation_utils import (
    apply_repetition_penalty,
    mask_out_of_tokenizer_vocab,
    prepare_generation_prompt,
    validate_sampling_parameters,
)


def _apply_top_k_top_p_np(logits: np.ndarray, temperature: float, top_k: int, top_p: float) -> np.ndarray:
    """Filter a [V] numpy logit vector with temperature, top-k, and top-p."""
    validate_sampling_parameters(temperature, top_k, top_p)
    if temperature == 0:
        greedy = int(np.argmax(logits))
        filtered = np.full_like(logits, -np.inf)
        filtered[greedy] = logits[greedy]
        return filtered
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
    stop_on_special_tokens: bool = True,
    ban_special_tokens: bool = True,
) -> list[int]:
    if max_new_tokens <= 0:
        return []
    validate_sampling_parameters(temperature, top_k, top_p)
    model.eval()
    generated: list[int] = []
    generated_seen: set[int] = set()
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

    # Reserve the requested output budget before cache warmup. For responses
    # longer than one context window, the cache is rebuilt from a sliding
    # window below, matching the Torch backend's overflow semantics.
    truncated = prepare_generation_prompt(
        prompt_ids,
        max_seq_len=model.config.max_seq_len,
        max_new_tokens=max_new_tokens,
    )
    context_tokens = list(truncated)
    idx = mx.array([truncated], dtype=mx.int32)
    logits, _, cache = model(idx, cache=None, cache_offset=0, return_cache=True)
    mx.eval(logits)
    cache_offset = idx.shape[1]

    for _ in range(max_new_tokens):
        # Pull just the last-step logits to host.
        last = np.asarray(logits[0, -1, :].astype(mx.float32))
        mask_out_of_tokenizer_vocab(
            last,
            int(getattr(tokenizer, "vocab_size", last.shape[-1])),
        )

        apply_repetition_penalty(last, generated_seen, repetition_penalty)

        if ban_special_tokens:
            for tok in banned:
                last[tok] = -np.inf

        filtered = _apply_top_k_top_p_np(last, temperature, top_k, top_p)
        token = _sample_host(filtered)
        if stop_on_special_tokens and token in stop_tokens:
            break
        generated.append(token)
        generated_seen.add(token)
        context_tokens.append(token)
        if len(generated) >= max_new_tokens:
            break

        if cache_offset + 1 <= model.config.max_seq_len:
            idx = mx.array([[token]], dtype=mx.int32)
            logits, _, cache = model(
                idx,
                cache=cache,
                cache_offset=cache_offset,
                return_cache=True,
            )
            cache_offset += 1
        else:
            # A KV cache cannot discard its oldest positions in place. Rebuild
            # it from the same right-aligned window Torch recomputes on its next
            # step, so generation continues rather than silently ending early.
            window = context_tokens[-model.config.max_seq_len :]
            idx = mx.array([window], dtype=mx.int32)
            logits, _, cache = model(idx, cache=None, cache_offset=0, return_cache=True)
            cache_offset = len(window)

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
