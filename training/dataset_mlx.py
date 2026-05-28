"""NumPy/MLX-native datasets and resumable batch iterator for the MLX backend."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.train_tokenizer import SpakieTokenizer


class PretrainDatasetMLX:
    """Memory-mapped numpy dataset for pretraining (returns numpy arrays).

    Sequences are non-overlapping `seq_len`-token windows packed back-to-back,
    matching standard LM pretraining (GPT-2/Llama style). The previous design
    treated every token offset as a separate sequence, which produced ~N
    indices for an N-token corpus and overflowed MLX's int32 dim cap on
    multi-billion-token corpora — preventing exact sampler resume.
    """

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.seq_len
        chunk = np.asarray(self.data[start : start + self.seq_len + 1], dtype=np.int32)
        return chunk[:-1], chunk[1:]


class ChatSFTDatasetMLX:
    """JSONL chat dataset with loss masking on non-assistant turns (numpy arrays)."""

    def __init__(self, jsonl_path: str, tokenizer: SpakieTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        messages = self.examples[idx]["messages"]
        input_ids: list[int] = []
        labels: list[int] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                role_token = self.tokenizer.system_id
            elif role == "user":
                role_token = self.tokenizer.user_id
            elif role == "assistant":
                role_token = self.tokenizer.assistant_id
            else:
                continue

            content_ids = self.tokenizer.encode(content)
            turn_ids = [role_token] + content_ids + [self.tokenizer.eos_id]
            if role == "assistant":
                turn_labels = [-100] + content_ids + [self.tokenizer.eos_id]
            else:
                turn_labels = [-100] * len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)

        input_ids = input_ids[: self.max_seq_len + 1]
        labels = labels[: self.max_seq_len + 1]

        x = input_ids[:-1]
        y = labels[1:]
        pad_len = self.max_seq_len - len(x)
        x = x + [self.tokenizer.pad_id] * pad_len
        y = y + [-100] * pad_len

        return np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32)


class PackedChatSFTDatasetMLX:
    """Pack chat examples into dense SFT sequences while preserving loss masks."""

    is_packed = True

    def __init__(self, jsonl_path: str, tokenizer: SpakieTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self._xs: list[np.ndarray] = []
        self._ys: list[np.ndarray] = []
        self._pack_examples()

    def _serialize_example(self, example: dict) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        labels: list[int] = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                role_token = self.tokenizer.system_id
            elif role == "user":
                role_token = self.tokenizer.user_id
            elif role == "assistant":
                role_token = self.tokenizer.assistant_id
            else:
                continue

            content_ids = self.tokenizer.encode(content)
            turn_ids = [role_token] + content_ids + [self.tokenizer.eos_id]
            if role == "assistant":
                turn_labels = [-100] + content_ids + [self.tokenizer.eos_id]
            else:
                turn_labels = [-100] * len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)
        return input_ids, labels

    def _append_pack(self, tokens: list[int], labels: list[int]) -> None:
        if len(tokens) < 2:
            return
        tokens = tokens[: self.max_seq_len + 1]
        labels = labels[: self.max_seq_len + 1]
        x = tokens[:-1]
        y = labels[1:]
        pad_len = self.max_seq_len - len(x)
        x = x + [self.tokenizer.pad_id] * pad_len
        y = y + [-100] * pad_len
        self._xs.append(np.asarray(x, dtype=np.int32))
        self._ys.append(np.asarray(y, dtype=np.int32))

    def _pack_examples(self) -> None:
        pack_tokens: list[int] = []
        pack_labels: list[int] = []
        limit = self.max_seq_len + 1
        for example in self.examples:
            tokens, labels = self._serialize_example(example)
            while tokens:
                remaining = limit - len(pack_tokens)
                if remaining <= 0:
                    self._append_pack(pack_tokens, pack_labels)
                    pack_tokens = []
                    pack_labels = []
                    remaining = limit
                take = min(remaining, len(tokens))
                pack_tokens.extend(tokens[:take])
                pack_labels.extend(labels[:take])
                tokens = tokens[take:]
                labels = labels[take:]
        if pack_tokens:
            self._append_pack(pack_tokens, pack_labels)

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._xs[idx], self._ys[idx]


def train_val_split_mlx(dataset, val_fraction: float = 0.05, seed: int = 42):
    """Deterministic grouped split. Returns (train_indices, val_indices).

    Synthetic SFT files often contain repeated prompts. Splitting by row leaks
    those duplicates into validation and makes early stopping overconfident, so
    chat datasets are grouped by their raw non-system messages before splitting.
    """
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    groups = _example_groups(dataset)
    if groups is not None:
        rng = np.random.default_rng(seed)
        group_ids = rng.permutation(len(groups)).tolist()
        val_indices = []
        train_indices = []
        for group_id in group_ids:
            target = val_indices if len(val_indices) < n_val else train_indices
            target.extend(groups[group_id])
        if train_indices and val_indices:
            return train_indices, val_indices

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def _raw_example_at(dataset, idx: int):
    if getattr(dataset, "is_packed", False):
        return None
    if hasattr(dataset, "examples"):
        return dataset.examples[idx]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _raw_example_at(dataset.dataset, int(dataset.indices[idx]))
    return None


def _example_signature(example: dict) -> str:
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            continue
        content = " ".join(str(msg.get("content", "")).lower().split())
        parts.append((role, content))
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def _example_groups(dataset) -> list[list[int]] | None:
    grouped: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        example = _raw_example_at(dataset, idx)
        if not isinstance(example, dict):
            return None
        grouped.setdefault(_example_signature(example), []).append(idx)
    return list(grouped.values())


class SubsetView:
    """Lightweight subset — mirrors torch.utils.data.Subset for dataset-agnostic code."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


class ResumableBatchSamplerMLX:
    """Deterministic shuffled sampler backed by numpy's Generator for exact resume."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
        rng_state: dict | None = None,
        indices: list[int] | np.ndarray | None = None,
        position: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.indices = (
            np.asarray(indices, dtype=np.int64) if indices is not None else np.empty(0, dtype=np.int64)
        )
        self.position = position
        if self.indices.size == 0:
            self._refresh_indices()

    def _refresh_indices(self) -> None:
        self.indices = self.rng.permutation(self.dataset_size).astype(np.int64, copy=False)
        self.position = 0

    def __iter__(self):
        while True:
            remaining = len(self.indices) - self.position
            if remaining < self.batch_size:
                if not self.drop_last and remaining > 0:
                    batch = self.indices[self.position :]
                    self._refresh_indices()
                    yield batch
                else:
                    self._refresh_indices()
                continue
            next_position = self.position + self.batch_size
            batch = self.indices[self.position : next_position]
            self.position = next_position
            yield batch

    def state_dict(self, *, copy_indices: bool = True) -> dict:
        return {
            "rng_state": self.rng.bit_generator.state,
            "indices": self.indices.copy() if copy_indices else self.indices,
            "position": self.position,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ResumableBatchSamplerMLX":
        return cls(
            dataset_size=int(state["dataset_size"]),
            batch_size=int(state["batch_size"]),
            drop_last=bool(state.get("drop_last", True)),
            rng_state=state.get("rng_state"),
            indices=state.get("indices"),
            position=int(state.get("position", 0)),
        )


def stack_batch(dataset, indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def iterate_batches(dataset, sampler: ResumableBatchSamplerMLX):
    """Yield stacked numpy batches from a sampler. Callers convert to mx.array."""
    for indices in sampler:
        yield stack_batch(dataset, indices)
