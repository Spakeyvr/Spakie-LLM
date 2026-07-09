from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from configs.default import (
    CHECKPOINT_CONFIG_SCHEMA_VERSION,
    SpakieConfig,
    config_from_dict,
    config_to_dict,
)
from model.transformer import SpakieGPT
from runtime.checkpoint_io import (
    UnsafeCheckpointError,
    load_mlx_checkpoint_config,
    load_mlx_model_weights_strict,
    load_torch_checkpoint,
)
from scripts.train import check_resume_optimizer, resume_sampler_mismatches
from training.pretrain import (
    ResumableBatchSampler,
    restore_rng_state,
    save_training_checkpoint,
)


class _MaliciousPayload:
    def __init__(self, marker: str):
        self.marker = marker

    def __reduce__(self):
        return os.system, (f"touch {self.marker}",)


class CheckpointIOTests(unittest.TestCase):
    @staticmethod
    def _tiny_config() -> SpakieConfig:
        return SpakieConfig(
            vocab_size=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            d_model=8,
            d_ff=16,
            max_seq_len=8,
            dropout=0.0,
            bias=False,
            pretrain_optimizer="adamw",
            pretrain_batch_size=2,
            pretrain_grad_accum_steps=1,
        )

    def test_safe_loader_rejects_pickle_code_without_side_effect(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "malicious.pt")
            marker = str(Path(temp_dir) / "executed")
            torch.save({"model": {}, "payload": _MaliciousPayload(marker)}, checkpoint_path)

            with self.assertRaises(UnsafeCheckpointError):
                load_torch_checkpoint(checkpoint_path)

            self.assertFalse(Path(marker).exists())

    def test_full_training_checkpoint_is_safe_loadable_and_restorable(self):
        config = self._tiny_config()
        model = SpakieGPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.optimizer_kind = "adamw"
        sampler = ResumableBatchSampler(8, 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = str(Path(temp_dir) / "roundtrip.pt")
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                global_step=3,
                tokens_processed=48,
                best_val_loss=1.5,
                val_loss=1.5,
                elapsed_time=2.0,
                train_sampler=sampler,
            )

            loaded = load_torch_checkpoint(checkpoint_path)
            restored_config = config_from_dict(loaded["config"])
            restore_rng_state(loaded["rng_state"])

        self.assertEqual(loaded["config_schema_version"], CHECKPOINT_CONFIG_SCHEMA_VERSION)
        self.assertEqual(restored_config, config)
        self.assertEqual(loaded["step"], 3)
        self.assertEqual(loaded["train_sampler"]["batch_size"], 2)

    def test_config_schema_rejects_truncated_payload(self):
        payload = config_to_dict(self._tiny_config())
        del payload["n_layers"]
        with self.assertRaisesRegex(ValueError, "missing fields: n_layers"):
            config_from_dict(payload)

    def test_mlx_metadata_roundtrip_and_legacy_refusal(self):
        config = self._tiny_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            weights = Path(temp_dir) / "model.safetensors"
            meta = {
                "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
                "config": config_to_dict(config),
            }
            Path(str(weights) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
            restored = load_mlx_checkpoint_config(str(weights))
            self.assertEqual(restored, config)

            Path(str(weights) + ".meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "predates full configuration"):
                load_mlx_checkpoint_config(str(weights))
            self.assertIsNone(
                load_mlx_checkpoint_config(str(weights), allow_legacy_config=True)
            )

    def test_strict_mlx_loader_does_not_accept_partial_weights(self):
        class FakeModel:
            def __init__(self):
                self.called = None

            def load_weights(self, weights, *, strict):
                self.called = (weights, strict)
                raise ValueError("missing 9 parameters")

        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            load_mlx_model_weights_strict(
                model, {"model.only.weight": object()}, path="partial.safetensors"
            )
        self.assertTrue(model.called[1])

    def test_sampler_batch_mismatch_is_detected(self):
        mismatch = resume_sampler_mismatches(
            {"dataset_size": 100, "batch_size": 128},
            dataset_size=100,
            batch_size=64,
        )
        self.assertEqual(mismatch, ["batch_size 128 -> 64"])

    def test_muon_hyperparameter_mismatch_requires_optimizer_reset(self):
        config = self._tiny_config()
        config.pretrain_optimizer = "muon"
        config.muon_ns_steps = 10
        state = {
            "meta": {
                "optimizer_kind": "muon",
                "muon_hyperparameters": {"ns_steps": 5},
            },
            "optimizer": {"state": object()},
            "_requested_config": config,
        }
        with self.assertRaises(SystemExit):
            check_resume_optimizer(
                state, "muon", backend="mlx", reset_optimizer=False
            )
        check_resume_optimizer(
            state, "muon", backend="mlx", reset_optimizer=True
        )
        self.assertNotIn("optimizer", state)


if __name__ == "__main__":
    unittest.main()
