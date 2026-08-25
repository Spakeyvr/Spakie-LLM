"""Exact Torch SFT interrupt/resume regression test."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime.backends import RuntimeSettings
from runtime.checkpoint_io import load_torch_checkpoint
from training.finetune import SFT_RESUME_SCHEMA_VERSION, finetune


class _TinySFTDataset(Dataset):
    def __init__(self, *, interrupt_on_call: int | None = None):
        self.examples = [
            {"messages": [
                {"role": "user", "content": f"u{idx}"},
                {"role": "assistant", "content": f"a{idx}"},
            ]}
            for idx in range(4)
        ]
        self.interrupt_on_call = interrupt_on_call
        self.calls = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        self.calls += 1
        if self.interrupt_on_call is not None and self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        x = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        y = torch.tensor([-100, 2, 3, -100], dtype=torch.long)
        return x, y


class TorchSFTResumeTests(unittest.TestCase):
    def test_interrupt_checkpoint_resumes_from_committed_optimizer_boundary(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"SPAKIE_MONITOR": "0"}
        ):
            config = SpakieConfig(
                vocab_size=32,
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                d_model=16,
                d_ff=32,
                swiglu_hidden=16,
                max_seq_len=4,
                dropout=0.0,
                bias=False,
                checkpoint_dir=tmpdir,
                sft_batch_size=1,
                sft_grad_accum_steps=2,
                sft_epochs=1,
                sft_optimizer="adamw",
            )
            runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
            val_dataset = _TinySFTDataset()

            finetune(
                SpakieGPT(config),
                _TinySFTDataset(interrupt_on_call=3),
                val_dataset,
                config,
                runtime,
                num_workers=0,
                interrupt_checkpoint_name="sft_interrupt.pt",
            )

            interrupt_path = Path(tmpdir) / "sft_interrupt.pt"
            state = load_torch_checkpoint(str(interrupt_path), map_location="cpu")
            self.assertEqual(state["sft_resume_schema_version"], SFT_RESUME_SCHEMA_VERSION)
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["epoch_microbatch_offset"], 2)
            self.assertIn("optimizer", state)
            self.assertIn("rng_state", state)
            self.assertIn("dataset_fingerprint", state["sft_resume_contract"])

            resumed_model = SpakieGPT(config)
            resumed_model.load_state_dict(state["model"])
            finetune(
                resumed_model,
                _TinySFTDataset(),
                val_dataset,
                config,
                runtime,
                num_workers=0,
                resume_state=state,
            )

            status = json.loads((Path(tmpdir) / "training_status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["step"], 2)


if __name__ == "__main__":
    unittest.main()
