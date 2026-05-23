import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime.backends import RuntimeSettings
from training.muon_core import (
    MUON_BF16_MAX_ABS,
    MUON_BF16_MAX_REL,
    is_muon_parameter_name,
)
from training.optimizers import MuonAdamW, configure_torch_optimizer, muon_newton_schulz_torch


class MuonCoreTests(unittest.TestCase):
    def test_newton_schulz_preserves_shape_and_is_finite(self):
        for shape in ((16, 16), (32, 8), (8, 32)):
            update = torch.randn(*shape)
            result = muon_newton_schulz_torch(update)
            self.assertEqual(tuple(result.shape), shape)
            self.assertTrue(torch.isfinite(result).all().item())

    def test_bf16_thresholds_are_concrete(self):
        self.assertEqual(MUON_BF16_MAX_ABS, 2e-2)
        self.assertEqual(MUON_BF16_MAX_REL, 5e-2)

    def test_parameter_name_routing(self):
        self.assertFalse(is_muon_parameter_name("tok_emb.weight", 2))
        self.assertFalse(is_muon_parameter_name("pos_emb.weight", 2))
        self.assertFalse(is_muon_parameter_name("blocks.0.ln1.weight", 1))
        self.assertTrue(is_muon_parameter_name("blocks.0.attn.qkv.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.fc1.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.gate_up.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.down.weight", 2))


class TorchMuonOptimizerTests(unittest.TestCase):
    def _config(self) -> SpakieConfig:
        return SpakieConfig(
            vocab_size=128,
            n_layers=2,
            n_heads=2,
            d_model=32,
            d_ff=64,
            max_seq_len=16,
            dropout=0.0,
            bias=False,
            pretrain_optimizer="muon",
        )

    def test_configure_defaults_to_hybrid_muon(self):
        config = self._config()
        model = SpakieGPT(config)
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        optimizer = configure_torch_optimizer(
            model,
            config,
            runtime,
            kind="muon",
            lr=1e-3,
            weight_decay=0.1,
        )
        self.assertIsInstance(optimizer, MuonAdamW)
        self.assertEqual(optimizer.optimizer_kind, "muon")
        self.assertIn("blocks.0.attn.qkv.weight", optimizer.muon_names)
        self.assertNotIn("tok_emb.weight", optimizer.muon_names)

    def test_one_muon_step_preserves_tied_embedding(self):
        torch.manual_seed(0)
        config = self._config()
        model = SpakieGPT(config)
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        optimizer = configure_torch_optimizer(
            model,
            config,
            runtime,
            kind="muon",
            lr=1e-3,
            weight_decay=0.1,
        )
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        self.assertIs(model.lm_head.weight, model.tok_emb.weight)

    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        config = self._config()
        model = SpakieGPT(config)
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        optimizer = configure_torch_optimizer(
            model,
            config,
            runtime,
            kind="muon",
            lr=1e-3,
            weight_decay=0.1,
        )
        idx = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        _, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        state = optimizer.state_dict()

        fresh = configure_torch_optimizer(
            model,
            config,
            runtime,
            kind="muon",
            lr=1e-3,
            weight_decay=0.1,
        )
        fresh.load_state_dict(state)
        self.assertEqual(fresh.optimizer_kind, "muon")
        self.assertTrue(fresh.state)


if __name__ == "__main__":
    unittest.main()
