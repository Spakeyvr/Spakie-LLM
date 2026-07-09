import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset_mlx import (
    ChatSFTDatasetMLX,
    HomogeneousStepSortedBatchSamplerMLX,
    LengthBucketBatchSamplerMLX,
    pack_sft_batch,
    PretokenizedChatSFTDatasetMLX,
    ResumableBatchSamplerMLX,
    sequence_lengths,
    StepSortedBatchSamplerMLX,
    trim_after_last_supervised_bucket,
    trim_right_padding_bucket,
    WindowSortedBatchSamplerMLX,
)


class FakeTokenizer:
    system_id = 10
    user_id = 11
    assistant_id = 12
    eos_id = 13
    pad_id = 0

    def encode(self, text):
        return [20 + (ord(ch) % 50) for ch in text]


class MLXDatasetUtilityTests(unittest.TestCase):
    def test_fixed_sampler_iteration_is_repeatable_and_does_not_advance_state(self):
        sampler = ResumableBatchSamplerMLX(
            dataset_size=11,
            batch_size=3,
            drop_last=False,
            seed=123,
        )
        before = sampler.state_dict()

        first = [batch.tolist() for batch in sampler.iter_fixed(max_batches=2)]
        second = [batch.tolist() for batch in sampler.iter_fixed(max_batches=2)]
        after = sampler.state_dict()

        self.assertEqual(first, second)
        self.assertEqual(before["position"], after["position"])
        np.testing.assert_array_equal(before["indices"], after["indices"])
        self.assertEqual(before["rng_state"], after["rng_state"])

    def test_pretokenized_chat_sft_dataset_matches_lazy_dataset(self):
        class Dataset(ChatSFTDatasetMLX):
            def __init__(self):
                self.tokenizer = FakeTokenizer()
                self.max_seq_len = 12
                self.examples = [
                    {
                        "messages": [
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "ok"},
                        ]
                    },
                    {
                        "messages": [
                            {"role": "system", "content": "brief"},
                            {"role": "user", "content": "?"},
                            {"role": "assistant", "content": "yes"},
                        ]
                    },
                ]

        lazy = Dataset()
        cached = PretokenizedChatSFTDatasetMLX(lazy)

        self.assertEqual(len(cached), len(lazy))
        self.assertEqual(cached.examples, lazy.examples)
        for idx in range(len(lazy)):
            x_lazy, y_lazy = lazy[idx]
            x_cached, y_cached = cached[idx]
            np.testing.assert_array_equal(x_cached, x_lazy)
            np.testing.assert_array_equal(y_cached, y_lazy)

    def test_trim_right_padding_bucket_preserves_nonpad_tokens(self):
        x = np.asarray(
            [
                [5, 6, 0, 0, 0, 0],
                [7, 8, 9, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        y = np.asarray(
            [
                [-100, 6, -100, -100, -100, -100],
                [-100, 8, 9, -100, -100, -100],
            ],
            dtype=np.int32,
        )

        tx, ty = trim_right_padding_bucket(x, y, bucket_multiple=2)

        self.assertEqual(tx.shape, (2, 4))
        self.assertEqual(ty.shape, (2, 4))
        np.testing.assert_array_equal(tx[:, :3], x[:, :3])
        np.testing.assert_array_equal(ty[:, :3], y[:, :3])

    def test_trim_after_last_supervised_bucket_drops_trailing_context(self):
        x = np.asarray(
            [
                [5, 6, 7, 8, 9, 0, 0, 0],
                [10, 11, 12, 13, 14, 15, 16, 0],
            ],
            dtype=np.int32,
        )
        y = np.asarray(
            [
                [-100, 6, 7, -100, -100, -100, -100, -100],
                [-100, 11, 12, 13, -100, -100, -100, -100],
            ],
            dtype=np.int32,
        )

        tx, ty = trim_after_last_supervised_bucket(x, y, bucket_multiple=2)

        self.assertEqual(tx.shape, (2, 4))
        self.assertEqual(ty.shape, (2, 4))
        np.testing.assert_array_equal(tx, x[:, :4])
        np.testing.assert_array_equal(ty, y[:, :4])

    def test_sequence_lengths_uses_input_or_label_content(self):
        class Dataset:
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                if idx == 0:
                    return (
                        np.asarray([1, 2, 0, 0], dtype=np.int32),
                        np.asarray([-100, -100, -100, -100], dtype=np.int32),
                    )
                return (
                    np.asarray([0, 0, 0, 0], dtype=np.int32),
                    np.asarray([-100, 3, 4, -100], dtype=np.int32),
                )

        np.testing.assert_array_equal(sequence_lengths(Dataset()), np.asarray([2, 3], dtype=np.int32))

    def test_length_bucket_sampler_groups_similar_lengths(self):
        lengths = np.asarray([5, 100, 6, 99, 7, 98, 8, 97], dtype=np.int32)
        sampler = LengthBucketBatchSamplerMLX(lengths, batch_size=2, bucket_size=8, seed=0)
        iterator = iter(sampler)
        batches = [next(iterator) for _ in range(4)]

        for batch in batches:
            spread = int(lengths[batch].max() - lengths[batch].min())
            self.assertLessEqual(spread, 2)

    def test_length_bucket_sampler_epoch_drops_only_global_tail(self):
        lengths = np.arange(23, dtype=np.int32)
        sampler = LengthBucketBatchSamplerMLX(
            lengths,
            batch_size=4,
            bucket_size=10,  # deliberately not batch-aligned
            drop_last=True,
            seed=7,
        )
        iterator = iter(sampler)

        first_epoch = [next(iterator) for _ in range(len(sampler))]
        first_indices = np.concatenate(first_epoch)

        self.assertEqual(len(sampler), 5)
        self.assertEqual(len(first_indices), 20)
        self.assertEqual(len(set(first_indices.tolist())), 20)
        # The very next batch belongs to the next permutation; an epoch-sized
        # consumer never has to spill into it to make up per-bucket tails.
        self.assertEqual(len(next(iterator)), 4)

    def test_step_sorted_sampler_preserves_optimizer_step_window(self):
        lengths = np.asarray([9, 1, 8, 2, 7, 3, 6, 4, 5, 10, 11, 12], dtype=np.int32)
        batch_size = 2
        grad_accum = 3
        plain = ResumableBatchSamplerMLX(len(lengths), batch_size, drop_last=True, seed=123)
        step_sorted = StepSortedBatchSamplerMLX(
            lengths,
            batch_size,
            grad_accum,
            drop_last=True,
            seed=123,
        )

        plain_iter = iter(plain)
        step_iter = iter(step_sorted)
        plain_window = np.concatenate([next(plain_iter) for _ in range(grad_accum)])
        sorted_window = np.concatenate([next(step_iter) for _ in range(grad_accum)])

        self.assertCountEqual(plain_window.tolist(), sorted_window.tolist())
        sorted_lengths = lengths[sorted_window]
        self.assertTrue(np.all(sorted_lengths[:-1] <= sorted_lengths[1:]))

    def test_window_sorted_sampler_preserves_larger_shuffle_window(self):
        lengths = np.asarray([9, 1, 8, 2, 7, 3, 6, 4, 5, 10, 11, 12], dtype=np.int32)
        batch_size = 2
        grad_accum = 2
        window_steps = 2
        plain = ResumableBatchSamplerMLX(len(lengths), batch_size, drop_last=True, seed=123)
        window_sorted = WindowSortedBatchSamplerMLX(
            lengths,
            batch_size,
            grad_accum,
            window_steps,
            drop_last=True,
            seed=123,
        )

        batches_per_window = grad_accum * window_steps
        plain_iter = iter(plain)
        sorted_iter = iter(window_sorted)
        plain_window = np.concatenate([next(plain_iter) for _ in range(batches_per_window)])
        sorted_batches = [next(sorted_iter) for _ in range(batches_per_window)]
        sorted_window = np.concatenate(sorted_batches)

        self.assertCountEqual(plain_window.tolist(), sorted_window.tolist())
        for batch in sorted_batches:
            spread = int(lengths[batch].max() - lengths[batch].min())
            self.assertLessEqual(spread, 3)

    def test_homogeneous_step_sorted_groups_same_rounded_shape(self):
        lengths = np.asarray(
            [
                1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 12, 13, 14, 15, 16,
            ],
            dtype=np.int32,
        )
        batch_size = 2
        grad_accum = 2
        sampler = HomogeneousStepSortedBatchSamplerMLX(
            lengths,
            batch_size,
            grad_accum,
            bucket_multiple=8,
            drop_last=True,
            seed=123,
        )

        iterator = iter(sampler)
        seen = []
        for _ in range(4):
            step_batches = [next(iterator) for _ in range(grad_accum)]
            rounded = [
                int(((lengths[batch].max() + 7) // 8) * 8)
                for batch in step_batches
            ]
            self.assertEqual(len(set(rounded)), 1)
            seen.extend(np.concatenate(step_batches).tolist())

        self.assertCountEqual(seen, list(range(len(lengths))))

    def test_pack_sft_batch_resets_segments_and_positions(self):
        x = np.asarray(
            [
                [10, 11, 12, 0, 0, 0],
                [20, 21, 0, 0, 0, 0],
                [30, 31, 32, 33, 0, 0],
            ],
            dtype=np.int32,
        )
        y = np.asarray(
            [
                [-100, 11, 12, -100, -100, -100],
                [-100, 21, -100, -100, -100, -100],
                [-100, 31, 32, 33, -100, -100],
            ],
            dtype=np.int32,
        )

        px, py, segments, positions = pack_sft_batch(x, y, max_seq_len=6)

        self.assertEqual(px.shape, (2, 6))
        self.assertEqual(int((segments >= 0).sum()), 9)
        np.testing.assert_array_equal(px[0], np.asarray([30, 31, 32, 33, 20, 21], dtype=np.int32))
        np.testing.assert_array_equal(py[0], np.asarray([-100, 31, 32, 33, -100, 21], dtype=np.int32))
        np.testing.assert_array_equal(segments[0], np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(positions[0], np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(px[1, :3], np.asarray([10, 11, 12], dtype=np.int32))
        np.testing.assert_array_equal(segments[1, :3], np.asarray([0, 0, 0], dtype=np.int32))
        np.testing.assert_array_equal(positions[1, :3], np.asarray([0, 1, 2], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
