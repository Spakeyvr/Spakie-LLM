from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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
    atomic_torch_save,
    discard_training_state,
    load_mlx_checkpoint_config,
    load_mlx_model_weights_strict,
    load_torch_checkpoint,
    validate_checkpoint_processed_data,
    validate_checkpoint_tokenizer,
)
from scripts.train import check_resume_optimizer, resume_sampler_mismatches
from scripts.run_pipeline import default_pretrain_checkpoint
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

    def test_atomic_torch_save_preserves_previous_checkpoint_on_failure(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            atomic_torch_save({"model": {"weight": torch.tensor([1])}}, str(path))
            original = path.read_bytes()

            def fail_after_partial(payload, temp_path):
                Path(temp_path).write_bytes(b"partial")
                raise OSError("disk full")

            with patch("torch.save", side_effect=fail_after_partial):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_torch_save({"model": {}}, str(path))
            self.assertEqual(path.read_bytes(), original)

    def test_inference_can_release_training_only_checkpoint_state(self):
        payload = {
            "model": {"weight": torch.tensor([1])},
            "config": config_to_dict(self._tiny_config()),
            "optimizer": {"large": torch.ones(16)},
            "train_sampler": {"indices": torch.arange(16)},
            "rng_state": {"torch": torch.get_rng_state()},
            "scaler": {"scale": 1.0},
        }
        discard_training_state(payload)
        self.assertEqual(set(payload), {"model", "config"})

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

    def test_actual_mlx_model_rejects_one_parameter_checkpoint(self):
        try:
            from mlx.utils import tree_flatten
            from model.transformer_mlx import SpakieGPTMLX
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f"MLX/Metal unavailable: {exc}")

        model = SpakieGPTMLX(self._tiny_config())
        name, value = tree_flatten(model.parameters())[0]
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            load_mlx_model_weights_strict(
                model, {f"model.{name}": value}, path="one-of-many.safetensors"
            )

    def test_sampler_batch_mismatch_is_detected(self):
        mismatch = resume_sampler_mismatches(
            {"dataset_size": 100, "batch_size": 128},
            dataset_size=100,
            batch_size=64,
        )
        self.assertEqual(mismatch, ["batch_size 128 -> 64"])

    def test_committed_cursor_is_not_advanced_by_torch_worker_prefetch(self):
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.arange(100), torch.arange(100))
        producer = ResumableBatchSampler(100, 2)
        committed = ResumableBatchSampler.from_state_dict(producer.state_dict())
        committed_iter = iter(committed)
        try:
            loader_iter = iter(DataLoader(dataset, batch_sampler=producer, num_workers=2))
        except RuntimeError as exc:
            if "Operation not permitted" in str(exc):
                self.skipTest(f"multiprocessing unavailable: {exc}")
            raise
        try:
            next(loader_iter)
            next(committed_iter)  # one optimizer-consumed microbatch
            self.assertEqual(committed.position, 2)
            self.assertGreater(producer.position, committed.position)
        finally:
            shutdown = getattr(loader_iter, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()

    def test_committed_cursor_is_not_advanced_by_mlx_prefetch_thread(self):
        import numpy as np

        from training.dataset_mlx import ResumableBatchSamplerMLX
        from training.prefetch_mlx import BatchPrefetcher

        class Dataset:
            def __getitem__(self, index):
                row = np.asarray([index], dtype=np.int64)
                return row, row

        producer = ResumableBatchSamplerMLX(100, 2)
        committed = ResumableBatchSamplerMLX.from_state_dict(
            producer.state_dict(copy_indices=False)
        )
        committed_iter = iter(committed)
        prefetcher = BatchPrefetcher(Dataset(), producer, maxsize=2)
        try:
            next(prefetcher)
            next(committed_iter)
            time.sleep(0.02)
            self.assertEqual(committed.position, 2)
            self.assertGreater(producer.position, committed.position)
        finally:
            prefetcher.close()

    def test_pipeline_consumes_final_checkpoint_not_stale_best(self):
        from argparse import Namespace

        path = default_pretrain_checkpoint(
            Namespace(preset="92m", backend="torch", smoke=False)
        )
        self.assertTrue(path.endswith("pretrain_final.pt"))

    def test_one_step_torch_run_publishes_rolling_and_final_checkpoints(self):
        from unittest.mock import patch

        from runtime.backends import RuntimeSettings
        from torch.utils.data import DataLoader, TensorDataset
        from training.pretrain import pretrain

        config = self._tiny_config()
        config.pretrain_max_steps = 1
        config.pretrain_target_tokens = 8
        config.pretrain_eval_interval = 99
        config.pretrain_checkpoint_interval = 1
        config.pretrain_eval_batches = 1
        x = torch.arange(32, dtype=torch.long).reshape(8, 4) % config.vocab_size
        dataset = TensorDataset(x, x.clone())
        sampler = ResumableBatchSampler(len(dataset), config.pretrain_batch_size)
        train_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=config.pretrain_batch_size)

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ, {"SPAKIE_MONITOR": "0"}
        ):
            config.checkpoint_dir = temp_dir
            pretrain(
                SpakieGPT(config),
                train_loader,
                val_loader,
                config,
                RuntimeSettings(torch.device("cpu"), "fp32"),
            )
            rolling = load_torch_checkpoint(str(Path(temp_dir) / "pretrain_interrupt.pt"))
            final = load_torch_checkpoint(str(Path(temp_dir) / "pretrain_final.pt"))

        self.assertEqual(rolling["step"], 1)
        self.assertEqual(final["step"], 1)
        self.assertEqual(final["train_sampler"]["position"], 2)

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

    def test_checkpoint_tokenizer_contract_must_match(self):
        saved = {"sha256": "expected", "vocab_size": 16}
        with patch(
            "runtime.checkpoint_io.tokenizer_contract",
            return_value={"sha256": "different", "vocab_size": 16},
        ):
            with self.assertRaisesRegex(ValueError, "different tokenizer"):
                validate_checkpoint_tokenizer(
                    {"tokenizer": saved}, "tokenizer.model", source="model.pt"
                )

    def test_legacy_checkpoint_requires_explicit_tokenizer_opt_in(self):
        with self.assertRaisesRegex(ValueError, "no tokenizer provenance"):
            validate_checkpoint_tokenizer(
                {}, "tokenizer.model", source="legacy.pt"
            )
        validate_checkpoint_tokenizer(
            {},
            "tokenizer.model",
            source="legacy.pt",
            allow_unverified=True,
        )

    def test_resume_rejects_different_processed_generation(self):
        with patch(
            "runtime.checkpoint_io.processed_manifest_sha256",
            return_value="current",
        ):
            with self.assertRaisesRegex(ValueError, "different processed-data"):
                validate_checkpoint_processed_data(
                    {"processed_data_manifest_sha256": "old"},
                    "processed",
                    source="resume.pt",
                )


if __name__ == "__main__":
    unittest.main()
