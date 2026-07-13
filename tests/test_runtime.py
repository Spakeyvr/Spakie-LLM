import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime.backends import (
    RuntimeSettings,
    autocast_context,
    create_grad_scaler,
    dataloader_kwargs,
    resolve_precision,
    resolve_runtime_settings,
)
import runtime.langid as langid


class RuntimeResolutionTests(unittest.TestCase):
    def test_mfa_varlen_rejects_grouped_query_attention(self):
        with self.assertRaisesRegex(ValueError, "does not support GQA"):
            SpakieConfig(
                n_heads=4,
                n_kv_heads=2,
                d_model=32,
                attention_backend="mfa-varlen",
            )

    def test_language_filter_fails_closed_without_model(self):
        text = "This is ordinary English prose with enough words to satisfy the " * 10
        with patch.object(langid, "load_langid_model", return_value=None):
            self.assertFalse(langid.is_probably_english(text))
            self.assertTrue(
                langid.is_probably_english(text, allow_heuristic_fallback=True)
            )

    def test_auto_device_prefers_mps_when_cuda_is_unavailable(self):
        with patch("runtime.backends.torch.cuda.is_available", return_value=False), patch(
            "runtime.backends.torch.backends.mps.is_available",
            return_value=True,
        ):
            runtime = resolve_runtime_settings("auto", "auto")
        self.assertEqual(runtime.device.type, "mps")

    def test_auto_precision_maps_to_bf16_on_mps_and_fp32_on_cpu(self):
        self.assertEqual(resolve_precision("auto", torch.device("mps")), "bf16")
        self.assertEqual(resolve_precision("auto", torch.device("cpu")), "fp32")

    def test_dataloader_uses_pin_memory_only_on_cuda(self):
        cuda_runtime = RuntimeSettings(device=torch.device("cuda"), precision="bf16")
        cpu_runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")

        cuda_kwargs = dataloader_kwargs(cuda_runtime, 2)
        cpu_kwargs = dataloader_kwargs(cpu_runtime, 2)

        self.assertTrue(cuda_kwargs["pin_memory"])
        self.assertFalse(cpu_kwargs["pin_memory"])
        self.assertTrue(cuda_kwargs["persistent_workers"])
        self.assertTrue(cpu_kwargs["persistent_workers"])

    def test_autocast_context_is_disabled_for_fp32(self):
        runtime = RuntimeSettings(device=torch.device("cpu"), precision="fp32")
        context = autocast_context(runtime)
        self.assertIsInstance(context, type(nullcontext()))

    def test_autocast_context_uses_actual_backend_device_type(self):
        runtime = RuntimeSettings(device=torch.device("mps"), precision="fp16")
        with patch("runtime.backends.torch.amp.autocast_mode.is_autocast_available", return_value=True), patch(
            "runtime.backends.torch.autocast"
        ) as autocast_mock:
            context = autocast_context(runtime)

        autocast_mock.assert_called_once_with(device_type="mps", dtype=torch.float16)
        self.assertIs(context, autocast_mock.return_value)

    def test_grad_scaler_is_enabled_only_for_fp16_autocast(self):
        fp16 = RuntimeSettings(device=torch.device("cpu"), precision="fp16")
        bf16 = RuntimeSettings(device=torch.device("cpu"), precision="bf16")
        with patch(
            "runtime.backends.torch.amp.autocast_mode.is_autocast_available",
            return_value=True,
        ):
            self.assertTrue(create_grad_scaler(fp16).is_enabled())
            self.assertFalse(create_grad_scaler(bf16).is_enabled())


class RuntimeSmokeTests(unittest.TestCase):
    def test_tiny_forward_and_loss_run_on_active_device(self):
        runtime = resolve_runtime_settings("auto", "auto")
        config = SpakieConfig(
            vocab_size=128,
            n_layers=2,
            n_heads=2,
            d_model=32,
            d_ff=64,
            max_seq_len=16,
            dropout=0.0,
        )

        model = SpakieGPT(config).to(runtime.device)
        idx = torch.randint(0, config.vocab_size, (2, 8), device=runtime.device)
        targets = torch.randint(0, config.vocab_size, (2, 8), device=runtime.device)

        with autocast_context(runtime):
            logits, loss = model(idx, targets)

        self.assertEqual(logits.shape, (2, 8, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss.detach().cpu()).item())


if __name__ == "__main__":
    unittest.main()
