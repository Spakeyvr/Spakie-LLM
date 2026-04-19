"""Runtime helpers for device, precision, and backend-aware execution."""

from runtime.backends import (
    DEVICE_CHOICES,
    PRECISION_CHOICES,
    RuntimeSettings,
    autocast_context,
    dataloader_kwargs,
    optimizer_kwargs,
    resolve_runtime_settings,
)

__all__ = [
    "DEVICE_CHOICES",
    "PRECISION_CHOICES",
    "RuntimeSettings",
    "autocast_context",
    "dataloader_kwargs",
    "optimizer_kwargs",
    "resolve_runtime_settings",
]
