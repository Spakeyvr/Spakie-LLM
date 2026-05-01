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

# MLX initializes Metal during import and can abort the interpreter on hosts
# without a usable Metal device. Keep MLX helpers behind explicit
# `runtime.mlx_backend` imports so generic CLI/help and PyTorch tests remain
# usable on non-Metal environments.
