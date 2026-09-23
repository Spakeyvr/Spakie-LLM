import random
import unittest

import numpy as np
import torch

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from scripts.train import seed_fresh_run
from training.pretrain import ResumableBatchSampler


class PretrainSeedTests(unittest.TestCase):
    def test_same_seed_repeats_initial_weights_and_different_seed_changes_them(self):
        cfg = SpakieConfig(vocab_size=128, n_layers=1, n_heads=2, n_kv_heads=1,
                           d_model=16, d_ff=32, swiglu_hidden=32, max_seq_len=8)
        def weights(seed):
            seed_fresh_run(seed, 'torch')
            return torch.cat([p.detach().flatten() for p in SpakieGPT(cfg).parameters()])
        first = weights(17)
        self.assertTrue(torch.equal(first, weights(17)))
        self.assertFalse(torch.equal(first, weights(29)))

    def test_resume_does_not_reset_rng(self):
        seed_fresh_run(17, 'torch')
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        seed_fresh_run(29, 'torch', resuming=True)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_seeded_sampler_repeats_and_saved_sampler_takes_precedence(self):
        def sampler(seed):
            return ResumableBatchSampler(100, 4, generator_state=torch.Generator().manual_seed(seed).get_state())
        original = sampler(17)
        other = sampler(17)
        self.assertTrue(torch.equal(original.indices, other.indices))
        self.assertFalse(torch.equal(original.indices, sampler(29).indices))
        next(iter(original))
        restored = ResumableBatchSampler.from_state_dict(original.state_dict())
        self.assertEqual(next(iter(original)), next(iter(restored)))


if __name__ == '__main__':
    unittest.main()
