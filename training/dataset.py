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

        # Truncate
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        # Pad
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train_val_split(dataset: Dataset, val_fraction: float = 0.05):
    """Sequential 95/5 split."""
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    return torch.utils.data.Subset(dataset, range(n_train)), torch.utils.data.Subset(dataset, range(n_train, n))
