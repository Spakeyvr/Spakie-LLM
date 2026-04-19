"""NumPy/MLX-native datasets and resumable batch iterator for the MLX backend."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.train_tokenizer import SpakieTokenizer


class PretrainDatasetMLX:
    """Memory-mapped numpy dataset for pretraining (returns numpy arrays)."""

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        chunk = np.asarray(self.data[idx : idx + self.seq_len + 1], dtype=np.int32)
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


def train_val_split_mlx(dataset, val_fraction: float = 0.05, seed: int = 42):
    """Deterministic random 95/5 split. Returns (train_indices, val_indices)."""
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return indices[n_val:].tolist(), indices[:n_val].tolist()


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
        indices: list[int] | None = None,
        position: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.indices = list(indices) if indices is not None else []
        self.position = position
        if not self.indices:
            self._refresh_indices()

    def _refresh_indices(self) -> None:
        self.indices = self.rng.permutation(self.dataset_size).tolist()
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

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.bit_generator.state,
            "indices": self.indices,
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


def stack_batch(dataset, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
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
