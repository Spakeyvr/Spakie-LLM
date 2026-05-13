"""Datasets for pretraining and SFT."""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from tokenizer.train_tokenizer import SpakieTokenizer


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


class ChatSFTDataset(Dataset):
    """JSONL chat dataset with loss masking on non-assistant turns."""

    def __init__(self, jsonl_path: str, tokenizer: SpakieTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]["messages"]
        input_ids = []
        labels = []

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
                # Loss on assistant content + eos, but not on the role token
                turn_labels = [-100] + content_ids + [self.tokenizer.eos_id]
            else:
                turn_labels = [-100] * len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)

        # Shift: input is all tokens except last, labels are all tokens except first
        # This way input_ids[i] predicts labels[i] = token at position i+1
        input_ids = input_ids[: self.max_seq_len + 1]
        labels = labels[: self.max_seq_len + 1]

        x = input_ids[:-1]
        y = labels[1:]

        # Pad
        pad_len = self.max_seq_len - len(x)
        x = x + [self.tokenizer.pad_id] * pad_len
        y = y + [-100] * pad_len

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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
