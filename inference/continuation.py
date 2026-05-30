"""Helpers for raw text continuation inference."""

from __future__ import annotations

from tokenizer.train_tokenizer import SpakieTokenizer


def decode_prefilled_continuation(
    tokenizer: SpakieTokenizer,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    show_special_tokens: bool = False,
) -> str:
    """Decode the visible continuation as prompt plus generated tokens."""
    return tokenizer.decode(
        [*prompt_ids, *response_ids],
        skip_special_tokens=not show_special_tokens,
    )
