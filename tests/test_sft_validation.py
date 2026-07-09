import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.backends import RuntimeSettings
from training.finetune import _evaluate_sft_loss


class _LossByMarkerModel(torch.nn.Module):
    def forward(self, x, y):
        return None, x[0, 0].float()


class TorchSFTValidationTests(unittest.TestCase):
    def test_validation_is_weighted_by_supervised_tokens(self):
        # Batch losses are 2 and 8, but their supervised-token counts are 1
        # and 3.  Equal batch weighting would incorrectly produce 5.
        batches = [
            (torch.tensor([[2]]), torch.tensor([[7, -100, -100]])),
            (torch.tensor([[8]]), torch.tensor([[1, 2, 3]])),
        ]
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        loss = _evaluate_sft_loss(_LossByMarkerModel(), batches, runtime)
        self.assertEqual(loss, 6.5)


if __name__ == "__main__":
    unittest.main()
