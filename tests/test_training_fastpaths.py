import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.dataset import PretrainDataset
from training.pretrain import ResumableBatchSampler


class TorchTrainingFastPathTests(unittest.TestCase):
    def test_pretrain_batched_fetch_matches_individual_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npy"
            np.save(path, np.arange(81, dtype=np.uint16))
            dataset = PretrainDataset(str(path), seq_len=8)
            indices = [4, 0, 7]

            expected_x = torch.stack([dataset[idx][0] for idx in indices])
            expected_y = torch.stack([dataset[idx][1] for idx in indices])
            actual_x, actual_y = next(
                iter(DataLoader(dataset, batch_sampler=[indices], num_workers=0))
            )

            self.assertTrue(torch.equal(actual_x, expected_x))
            self.assertTrue(torch.equal(actual_y, expected_y))
            self.assertTrue(actual_x.is_contiguous())
            self.assertTrue(actual_y.is_contiguous())

    def test_resumable_sampler_fast_advance_matches_iteration(self):
        for drop_last in (False, True):
            with self.subTest(drop_last=drop_last):
                iterated = ResumableBatchSampler(
                    dataset_size=11,
                    batch_size=4,
                    drop_last=drop_last,
                )
                advanced = ResumableBatchSampler.from_state_dict(
                    iterated.state_dict()
                )
                iterator = iter(iterated)
                for _ in range(7):
                    next(iterator)
                advanced.advance_batches(7)

                self.assertEqual(iterated.position, advanced.position)
                self.assertTrue(torch.equal(iterated.indices, advanced.indices))
                self.assertTrue(
                    torch.equal(
                        iterated.generator.get_state(),
                        advanced.generator.get_state(),
                    )
                )
                self.assertEqual(next(iterator), next(iter(advanced)))


if __name__ == "__main__":
    unittest.main()
