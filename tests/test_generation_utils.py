"""Tests for generation behavior shared across Torch and MLX."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.generation_utils import (
    apply_repetition_penalty,
    mask_out_of_tokenizer_vocab,
    prepare_generation_prompt,
    prompt_token_budget,
)
from configs.default import SpakieConfig
from inference.generate import generate as generate_torch
from model.transformer import SpakieGPT


class GenerationUtilityTests(unittest.TestCase):
    def test_torch_generation_uses_sliding_context_for_long_response(self):
        class Tokenizer:
            eos_id = 120
            user_id = 121
            assistant_id = 122
            system_id = 123
            json_id = 124
            pad_id = 125

        config = SpakieConfig(
            vocab_size=128,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_model=16,
            d_ff=32,
            swiglu_hidden=16,
            max_seq_len=8,
            dropout=0.0,
            bias=False,
        )
        model = SpakieGPT(config)

        generated = generate_torch(
            model,
            Tokenizer(),
            prompt_ids=[1, 2, 3, 4, 5, 6, 7],
            max_new_tokens=12,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            stop_on_special_tokens=False,
            ban_special_tokens=False,
        )

        self.assertEqual(len(generated), 12)

    def test_prompt_reserves_actual_requested_output_budget(self):
        self.assertEqual(prompt_token_budget(8, 4), 4)
        self.assertEqual(
            prepare_generation_prompt(range(7), max_seq_len=8, max_new_tokens=4),
            [3, 4, 5, 6],
        )
        self.assertEqual(prompt_token_budget(8, 99), 1)

    def test_repetition_penalty_applies_once_per_unique_token_numpy(self):
        logits = np.asarray([4.0, -4.0, 1.0], dtype=np.float32)
        apply_repetition_penalty(logits, [0, 0, 1, 1, 1], 2.0)
        np.testing.assert_array_equal(logits, np.asarray([2.0, -8.0, 1.0]))

    def test_repetition_penalty_has_identical_torch_semantics(self):
        logits = torch.tensor([4.0, -4.0, 1.0])
        apply_repetition_penalty(logits, [0, 0, 1, 1, 1], 2.0)
        torch.testing.assert_close(logits, torch.tensor([2.0, -8.0, 1.0]))

    def test_repetition_penalty_rejects_nonpositive_value(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            apply_repetition_penalty(np.ones(2), [0], 0.0)

    def test_model_head_wider_than_tokenizer_masks_undecodable_ids(self):
        numpy_logits = np.arange(6, dtype=np.float32)
        mask_out_of_tokenizer_vocab(numpy_logits, 4)
        np.testing.assert_array_equal(numpy_logits[:4], np.arange(4, dtype=np.float32))
        self.assertTrue(np.isneginf(numpy_logits[4:]).all())

        torch_logits = torch.arange(6, dtype=torch.float32)
        mask_out_of_tokenizer_vocab(torch_logits, 4)
        self.assertTrue(torch.isneginf(torch_logits[4:]).all())

    def test_tokenizer_wider_than_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds model vocabulary"):
            mask_out_of_tokenizer_vocab(np.ones(4), 5)


if __name__ == "__main__":
    unittest.main()
