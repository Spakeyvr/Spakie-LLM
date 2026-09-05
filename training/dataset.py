"""Datasets for pretraining and SFT."""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.train_tokenizer import SpakieTokenizer
from training.sft_tokenization import (
    encode_sft_example,
    pad_sft_example,
    prepare_sft_examples,
)


class PretrainDataset(Dataset):
    """Memory-mapped numpy dataset for pretraining."""

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y

    def __getitems__(self, indices):
        """Fetch a batch with one allocation and one corpus read per row."""
        batch = np.empty((len(indices), self.seq_len + 1), dtype=np.int64)
        for row, idx in enumerate(indices):
            start = int(idx) * self.seq_len
            batch[row] = self.data[start : start + self.seq_len + 1]

        tokens = torch.from_numpy(batch)
        return list(zip(tokens[:, :-1], tokens[:, 1:]))


class ChatSFTDataset(Dataset):
    """JSONL chat dataset with loss masking on non-assistant turns."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: SpakieTokenizer,
        max_seq_len: int,
        *,
        pretokenize: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.examples, self._encoded, self.validation_stats = prepare_sft_examples(
            self.examples,
            self.tokenizer,
            self.max_seq_len,
            keep_cache=pretokenize,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoded = (
            getattr(self, "_encoded", None)[idx]
            if getattr(self, "_encoded", None) is not None
            else encode_sft_example(self.examples[idx], self.tokenizer, self.max_seq_len)
        )
        x, y = pad_sft_example(
            encoded,
            max_seq_len=self.max_seq_len,
            pad_id=self.tokenizer.pad_id,
        )
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    def sequence_lengths(self) -> np.ndarray:
        if getattr(self, "_encoded", None) is not None:
            return np.asarray([item.length for item in self._encoded], dtype=np.int32)
        return np.asarray(
            [
                encode_sft_example(example, self.tokenizer, self.max_seq_len).length
                for example in self.examples
            ],
            dtype=np.int32,
        )


def train_val_split(dataset: Dataset, val_fraction: float = 0.05, seed: int = 42):
    """Deterministic grouped 95/5 split for chat datasets."""
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
            return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)


def _raw_example_at(dataset, idx: int):
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
        if role == "assistant" and msg.get("train", True) is not False:
            content = "<supervised-target>"
        else:
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
