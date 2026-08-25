"""Backend-neutral SFT rendering, validation, and compact token caching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


def _raw_sft_example_at(dataset, idx: int):
    if hasattr(dataset, "examples"):
        return dataset.examples[idx]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _raw_sft_example_at(dataset.dataset, int(dataset.indices[idx]))
    return None


def sft_dataset_fingerprint(dataset) -> str:
    """Hash the ordered logical SFT split used by an exact-resume checkpoint."""
    digest = hashlib.sha256()
    digest.update(f"rows:{len(dataset)}\n".encode())
    for idx in range(len(dataset)):
        example = _raw_sft_example_at(dataset, idx)
        if example is None:
            # Generic fallback for tests/custom datasets without raw JSON rows.
            x, y = dataset[idx]
            for value in (x, y):
                array = np.asarray(value, dtype=np.int64)
                digest.update(array.shape.__repr__().encode())
                digest.update(array.tobytes())
            continue
        digest.update(
            json.dumps(
                example, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class EncodedSFTExample:
    """Unpadded shifted input/label arrays for one chat example."""

    x: np.ndarray
    y: np.ndarray

    @property
    def length(self) -> int:
        return int(self.x.shape[0])


class SFTSequenceTooLongError(ValueError):
    def __init__(self, rendered_tokens: int, max_rendered_tokens: int):
        self.rendered_tokens = int(rendered_tokens)
        self.max_rendered_tokens = int(max_rendered_tokens)
        super().__init__(
            f"rendered SFT example has {rendered_tokens} tokens; maximum is "
            f"{max_rendered_tokens}. Rebuild/filter the data instead of truncating "
            "an assistant response."
        )


class SFTNoSupervisedTokensError(ValueError):
    pass


def encode_sft_example(example: dict, tokenizer, max_seq_len: int) -> EncodedSFTExample:
    """Render one chat example without padding and reject lossy truncation."""
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in example.get("messages", []):
        role = msg.get("role")
        if role == "system":
            role_token = tokenizer.system_id
        elif role == "user":
            role_token = tokenizer.user_id
        elif role == "assistant":
            role_token = tokenizer.assistant_id
        else:
            continue

        content_ids = tokenizer.encode(str(msg.get("content", "")))
        turn_ids = [role_token] + content_ids + [tokenizer.eos_id]
        if role == "assistant" and msg.get("train", True):
            turn_labels = [-100] + content_ids + [tokenizer.eos_id]
        else:
            turn_labels = [-100] * len(turn_ids)
        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    max_rendered_tokens = max_seq_len + 1
    if len(input_ids) > max_rendered_tokens:
        raise SFTSequenceTooLongError(len(input_ids), max_rendered_tokens)

    x = np.asarray(input_ids[:-1], dtype=np.int32)
    y = np.asarray(labels[1:], dtype=np.int32)
    if y.size == 0 or not np.any(y != -100):
        raise SFTNoSupervisedTokensError(
            "SFT example has no supervised assistant content after rendering"
        )
    return EncodedSFTExample(x=x, y=y)


def pad_sft_example(
    encoded: EncodedSFTExample,
    *,
    max_seq_len: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad one compact encoded example to the model context length."""
    if encoded.length > max_seq_len:
        raise ValueError("encoded SFT example exceeds max_seq_len")
    x = np.full((max_seq_len,), pad_id, dtype=np.int32)
    y = np.full((max_seq_len,), -100, dtype=np.int32)
    x[: encoded.length] = encoded.x
    y[: encoded.length] = encoded.y
    return x, y


def prepare_sft_examples(
    examples: list[dict],
    tokenizer,
    max_seq_len: int,
    *,
    keep_cache: bool,
) -> tuple[list[dict], list[EncodedSFTExample] | None, dict[str, int]]:
    """Validate all rows, drop unusable rows, and optionally retain compact tokens."""
    valid_examples: list[dict] = []
    encoded_cache: list[EncodedSFTExample] | None = [] if keep_cache else None
    stats = {"overlength": 0, "no_supervised_tokens": 0}
    for example in examples:
        try:
            encoded = encode_sft_example(example, tokenizer, max_seq_len)
        except SFTSequenceTooLongError:
            stats["overlength"] += 1
            continue
        except SFTNoSupervisedTokensError:
            stats["no_supervised_tokens"] += 1
            continue
        valid_examples.append(example)
        if encoded_cache is not None:
            encoded_cache.append(encoded)
    return valid_examples, encoded_cache, stats
