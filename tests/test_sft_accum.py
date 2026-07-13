import unittest

import torch

from configs.default import SpakieConfig
from training.finetune import _scale_partial_sft_grads


class TorchSFTAccumulationTests(unittest.TestCase):
    def test_partial_tail_gradients_are_rescaled_to_actual_microbatch_count(self):
        config = SpakieConfig()
        config.sft_grad_accum_steps = 4
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[0.5]])

        _scale_partial_sft_grads(model, config, microbatches_in_step=2)

        self.assertTrue(torch.allclose(model.weight.grad, torch.tensor([[1.0]])))

    def test_full_accumulation_window_keeps_gradients_unchanged(self):
        config = SpakieConfig()
        config.sft_grad_accum_steps = 4
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.tensor([[1.25]])

        _scale_partial_sft_grads(model, config, microbatches_in_step=4)

        self.assertTrue(torch.allclose(model.weight.grad, torch.tensor([[1.25]])))


class MLXSFTAccumulationTests(unittest.TestCase):
    def test_partial_tail_gradient_tree_is_rescaled(self):
        try:
            import mlx.core as mx
            from training.finetune_mlx import _scale_partial_accum_grads
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"MLX is unavailable: {exc}")

        config = SpakieConfig()
        config.sft_grad_accum_steps = 4
        grads = {
            "a": mx.array([0.5], dtype=mx.float32),
            "nested": {"b": mx.array([1.0], dtype=mx.float32)},
        }

        scaled = _scale_partial_accum_grads(grads, config, micro_in_step=2)
        mx.eval(scaled)

        self.assertEqual(float(scaled["a"].item()), 1.0)
        self.assertEqual(float(scaled["nested"]["b"].item()), 2.0)


if __name__ == "__main__":
    unittest.main()
