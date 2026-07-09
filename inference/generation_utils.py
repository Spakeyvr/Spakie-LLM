"""Backend-neutral generation contracts shared by Torch and MLX."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


Logits = TypeVar("Logits")


def prompt_token_budget(max_seq_len: int, max_new_tokens: int) -> int:
    """Return prompt capacity after reserving the requested response budget.

    At least one prompt token is retained. Responses longer than the context
    window are still supported by each backend's sliding-window generation.
    """
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    reserved = min(max(max_new_tokens, 0), max_seq_len - 1)
    return max(1, max_seq_len - reserved)


def prepare_generation_prompt(
    prompt_ids: Iterable[int],
    *,
    max_seq_len: int,
    max_new_tokens: int,
) -> list[int]:
    """Crop a prompt consistently while reserving requested output capacity."""
    prompt = list(prompt_ids)
    if not prompt:
        raise ValueError("generation requires at least one prompt token")
    return prompt[-prompt_token_budget(max_seq_len, max_new_tokens) :]


def apply_repetition_penalty(
    logits: Logits,
    generated_token_ids: Iterable[int],
    penalty: float,
) -> Logits:
    """Apply one repetition penalty per unique generated token, in place.

    This follows the common sign-aware definition and intentionally does not
    compound the penalty when a token has appeared multiple times.
    ``logits`` may be a NumPy vector or a one-dimensional Torch tensor.
    """
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if penalty == 1.0:
        return logits
    for token_id in set(generated_token_ids):
        value = logits[token_id]
        logits[token_id] = value / penalty if value > 0 else value * penalty
    return logits


def mask_out_of_tokenizer_vocab(logits: Logits, tokenizer_vocab_size: int) -> Logits:
    """Prevent a model head wider than its tokenizer from emitting invalid IDs."""
    if tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
    model_vocab_size = int(logits.shape[-1])
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError(
            f"tokenizer vocabulary ({tokenizer_vocab_size}) exceeds model vocabulary "
            f"({model_vocab_size})"
        )
    if tokenizer_vocab_size < model_vocab_size:
        logits[..., tokenizer_vocab_size:] = -float("inf")
    return logits
