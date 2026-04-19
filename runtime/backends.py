"""Backend-aware runtime helpers for CUDA, MPS, and CPU."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")
PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


@dataclass(frozen=True)
class RuntimeSettings:
    device: torch.device
    precision: str

    @property
    def autocast_enabled(self) -> bool:
        return self.precision != "fp32" and torch.amp.autocast_mode.is_autocast_available(self.device.type)

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return None

    @property
    def description(self) -> str:
        return f"Device: {self.device.type} | Precision: {self.precision}"


def _mps_is_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def resolve_device(device_name: str) -> torch.device:
    requested = (device_name or "auto").lower()
    if requested not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device '{device_name}'. Choices: {', '.join(DEVICE_CHOICES)}")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not _mps_is_available():
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")

    return torch.device("cpu")


def resolve_precision(precision_name: str, device: torch.device) -> str:
    requested = (precision_name or "auto").lower()
    if requested not in PRECISION_CHOICES:
        raise ValueError(f"Unsupported precision '{precision_name}'. Choices: {', '.join(PRECISION_CHOICES)}")

    if requested != "auto":
        return requested

    if device.type == "cuda":
        return "bf16"
    if device.type == "mps":
        return "bf16"
    return "fp32"


def resolve_runtime_settings(device_name: str = "auto", precision_name: str = "auto") -> RuntimeSettings:
    device = resolve_device(device_name)
    precision = resolve_precision(precision_name, device)
    return RuntimeSettings(device=device, precision=precision)


def autocast_context(runtime: RuntimeSettings):
    if not runtime.autocast_enabled:
        return nullcontext()
    return torch.autocast(device_type=runtime.device.type, dtype=runtime.autocast_dtype)


def dataloader_kwargs(runtime: RuntimeSettings, num_workers: int) -> dict[str, object]:
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    return {
        "num_workers": num_workers,
        "pin_memory": runtime.device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }


def optimizer_kwargs(runtime: RuntimeSettings) -> dict[str, object]:
    return {
        "fused": runtime.device.type == "cuda",
    }
