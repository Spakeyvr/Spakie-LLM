import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
    MUON_NS_COEFFICIENTS,
    MUON_NS_POLISH_COEFFICIENTS,
    MuonPrecomputeError,
    MuonSettings,
    is_muon_parameter_name,
    muon_ns_step_coefficients,
    muon_projection_split_count,
    muon_settings_from_config,
    muon_update_split_count,
    should_adamw_fallback,
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
        self.assertFalse(is_muon_parameter_name("blocks.0.ln1.weight", 1))
        self.assertTrue(is_muon_parameter_name("blocks.0.attn.qkv.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.fc1.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.gate_up.weight", 2))
        self.assertTrue(is_muon_parameter_name("blocks.0.mlp.down.weight", 2))
        self.assertEqual(muon_projection_split_count("blocks.0.attn.qkv.weight"), 3)
        self.assertEqual(muon_projection_split_count("blocks.0.attn.kv_proj.weight"), 2)

    def test_ns_schedule_places_polish_iterations_last(self):
        self.assertEqual(muon_ns_step_coefficients(10), (MUON_NS_COEFFICIENTS,) * 10)
        schedule = muon_ns_step_coefficients(10, MUON_NS_COEFFICIENTS, 2)
        self.assertEqual(schedule[:8], (MUON_NS_COEFFICIENTS,) * 8)
        self.assertEqual(schedule[8:], (MUON_NS_POLISH_COEFFICIENTS,) * 2)

    def test_ns_schedule_rejects_out_of_range_polish(self):
        for polish in (-1, 11):
            with self.assertRaises(ValueError):
                muon_ns_step_coefficients(10, MUON_NS_COEFFICIENTS, polish)
        with self.assertRaises(ValueError):
            SpakieConfig(muon_ns_steps=5, muon_ns_polish_steps=6)
        config = SpakieConfig()
        config.muon_ns_polish_steps = config.muon_ns_steps + 1
        with self.assertRaises(ValueError):
            muon_settings_from_config(config)

    def test_polish_iterations_pin_singular_values_at_one(self):
        # Heavy-tailed spectrum (1000:1), like real momentum matrices.
        generator = torch.Generator().manual_seed(0)
        u, _ = torch.linalg.qr(torch.randn(64, 64, generator=generator, dtype=torch.float64))
        v, _ = torch.linalg.qr(torch.randn(192, 64, generator=generator, dtype=torch.float64))
        update = ((u * torch.logspace(0, -3, 64, dtype=torch.float64)) @ v.T).float()

        quintic = torch.linalg.svdvals(muon_newton_schulz_torch(update, ns_steps=10).double())
        paper = torch.linalg.svdvals(
            muon_newton_schulz_torch(update, ns_steps=10, ns_polish_steps=2).double()
        )

        self.assertGreater(float((quintic - 1).abs().max()), 0.1)
        self.assertLess(float((paper - 1).abs().max()), 1e-2)

    def test_gate_up_split_follows_its_own_setting(self):
        gate_up = "blocks.0.mlp.gate_up.weight"
        qkv = "blocks.0.attn.qkv.weight"
        self.assertEqual(muon_projection_split_count(gate_up), 2)
        self.assertEqual(muon_update_split_count(gate_up, MuonSettings()), 1)
        self.assertEqual(muon_update_split_count(gate_up, MuonSettings(split_gate_up=True)), 2)
        self.assertEqual(
            muon_update_split_count(gate_up, MuonSettings(qkv_split=False, split_gate_up=True)), 2
        )
        self.assertEqual(muon_update_split_count(qkv, MuonSettings(split_gate_up=True)), 3)
        self.assertEqual(muon_update_split_count(qkv, MuonSettings(qkv_split=False)), 1)
        self.assertEqual(muon_update_split_count("blocks.0.mlp.down.weight", MuonSettings(split_gate_up=True)), 1)

    def test_runtime_fallback_is_only_allowed_before_mutation(self):
        optimizer = type("Optimizer", (), {"optimizer_kind": "muon"})()
        config = SpakieConfig(pretrain_optimizer="muon")
        self.assertTrue(
            should_adamw_fallback(
                MuonPrecomputeError("safe"), optimizer, config, stage="pretrain", allow=True
            )
        )
        self.assertFalse(
            should_adamw_fallback(
                RuntimeError("unknown mutation state"),
                optimizer,
                config,
                stage="pretrain",
                allow=True,
            )
        )


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

    def test_newton_schulz_failure_does_not_partially_update_parameters(self):
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
        before = {name: param.detach().clone() for name, param in model.named_parameters()}

        with patch(
            "training.optimizers.muon_newton_schulz_torch",
            side_effect=RuntimeError("forced Newton-Schulz failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced Newton-Schulz failure"):
                optimizer.step()

        for name, param in model.named_parameters():
            self.assertTrue(torch.equal(param, before[name]), name)
        self.assertFalse(optimizer.state)

    def test_fused_kv_projection_is_split_into_two_muon_updates(self):
        config = self._config()
        model = SpakieGPT(config)
        optimizer = configure_torch_optimizer(
            model,
            config,
            RuntimeSettings(device=torch.device("cpu"), precision="fp32"),
            kind="muon",
            lr=1e-3,
            weight_decay=0.1,
        )
        chunks = list(
            optimizer._update_chunks(
                "blocks.0.attn.kv_proj.weight", torch.zeros(12, 8)
            )
        )
        self.assertEqual([start for start, _ in chunks], [0, 6])
        self.assertEqual([tuple(chunk.shape) for _, chunk in chunks], [(6, 8), (6, 8)])

    def test_fused_gate_up_is_split_only_when_enabled(self):
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        for split, expected in ((False, [(0, (64, 32))]), (True, [(0, (32, 32)), (32, (32, 32))])):
            config = self._config()
            config.muon_split_gate_up = split
            optimizer = configure_torch_optimizer(
                SpakieGPT(config), config, runtime, kind="muon", lr=1e-3, weight_decay=0.1
            )
            chunks = list(optimizer._update_chunks("blocks.0.mlp.gate_up.weight", torch.zeros(64, 32)))
            self.assertEqual([(start, tuple(chunk.shape)) for start, chunk in chunks], expected)


if __name__ == "__main__":
    unittest.main()
