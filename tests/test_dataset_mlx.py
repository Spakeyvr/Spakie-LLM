import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dataset_mlx import (
    LengthBucketBatchSamplerMLX,
    sequence_lengths,
    trim_right_padding_bucket,
)


class MLXDatasetUtilityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
