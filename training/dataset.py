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
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)
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
    """Deterministic random 95/5 split."""
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)
