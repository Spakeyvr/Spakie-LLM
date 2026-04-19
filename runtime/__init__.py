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

try:
    from runtime.mlx_backend import (
        MLX_PRECISION_CHOICES,
        MLXRuntimeSettings,
        resolve_mlx_runtime,
        save_safetensors,
        load_safetensors,
        save_meta_json,
        load_meta_json,
    )

    __all__ += [
        "MLX_PRECISION_CHOICES",
        "MLXRuntimeSettings",
        "resolve_mlx_runtime",
        "save_safetensors",
        "load_safetensors",
        "save_meta_json",
        "load_meta_json",
    ]
except ImportError:
    pass
