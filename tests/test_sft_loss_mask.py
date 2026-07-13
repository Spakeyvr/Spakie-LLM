"""Regression tests for per-message SFT supervision metadata."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset import ChatSFTDataset, train_val_split
from training.dataset_mlx import ChatSFTDatasetMLX, train_val_split_mlx


class FakeTokenizer:
    system_id = 10
    user_id = 11
    assistant_id = 12
    eos_id = 13
    pad_id = 0

    def encode(self, text: str) -> list[int]:
        return [20 + ord(char) for char in text]


def _dataset_without_file(dataset_type, messages):
    dataset = object.__new__(dataset_type)
    dataset.tokenizer = FakeTokenizer()
    dataset.max_seq_len = 32
    dataset.examples = [{"messages": messages}]
    return dataset


class SFTLossMaskTests(unittest.TestCase):
    def _assert_only_final_assistant_is_supervised(self, dataset_type) -> None:
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "train": False},
            {"role": "user", "content": "v"},
            {"role": "assistant", "content": "b", "train": True},
        ]
        _, labels = _dataset_without_file(dataset_type, messages)[0]
        labels = np.asarray(labels)

        supervised = labels[labels != -100].tolist()
        self.assertEqual(supervised, FakeTokenizer().encode("b") + [FakeTokenizer.eos_id])

    def _assert_ordinary_assistant_turns_remain_supervised(self, dataset_type) -> None:
        messages = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "v"},
            {"role": "assistant", "content": "b"},
        ]
        _, labels = _dataset_without_file(dataset_type, messages)[0]
        labels = np.asarray(labels)

        supervised = labels[labels != -100].tolist()
        expected = (
            FakeTokenizer().encode("a")
            + [FakeTokenizer.eos_id]
            + FakeTokenizer().encode("b")
            + [FakeTokenizer.eos_id]
        )
        self.assertEqual(supervised, expected)

    def test_torch_dataset_honors_explicit_assistant_training_mask(self):
        self._assert_only_final_assistant_is_supervised(ChatSFTDataset)

    def test_mlx_dataset_honors_explicit_assistant_training_mask(self):
        self._assert_only_final_assistant_is_supervised(ChatSFTDatasetMLX)

    def test_torch_dataset_defaults_to_all_assistant_turns(self):
        self._assert_ordinary_assistant_turns_remain_supervised(ChatSFTDataset)

    def test_mlx_dataset_defaults_to_all_assistant_turns(self):
        self._assert_ordinary_assistant_turns_remain_supervised(ChatSFTDatasetMLX)

    def test_same_prompt_with_different_targets_stays_in_one_split(self):
        class RawDataset:
            examples = [
                {"messages": [
                    {"role": "user", "content": "Explain gravity"},
                    {"role": "assistant", "content": "Answer A"},
                ]},
                {"messages": [
                    {"role": "user", "content": "Explain gravity"},
                    {"role": "assistant", "content": "Answer B"},
                ]},
                {"messages": [
                    {"role": "user", "content": "Explain photosynthesis"},
                    {"role": "assistant", "content": "Answer C"},
                ]},
            ]

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                return self.examples[index]

        dataset = RawDataset()
        torch_train, torch_val = train_val_split(dataset, val_fraction=0.34, seed=1)
        mlx_train, mlx_val = train_val_split_mlx(dataset, val_fraction=0.34, seed=1)
        self.assertEqual(0 in torch_train.indices, 1 in torch_train.indices)
        self.assertEqual(0 in torch_val.indices, 1 in torch_val.indices)
        self.assertEqual(0 in mlx_train, 1 in mlx_train)
        self.assertEqual(0 in mlx_val, 1 in mlx_val)


if __name__ == "__main__":
    unittest.main()
