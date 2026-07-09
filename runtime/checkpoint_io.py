"""Checkpoint metadata and safe PyTorch deserialization helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from configs.default import (
    CHECKPOINT_CONFIG_SCHEMA_VERSION,
    SpakieConfig,
    config_from_dict,
)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be read by PyTorch's restricted loader."""


def load_torch_checkpoint(
    path: str,
    *,
    map_location: Any = "cpu",
    trust_checkpoint: bool = False,
) -> dict:
    """Load a checkpoint without executing arbitrary pickle globals by default.

    Legacy checkpoints containing custom Python objects can still be loaded, but
    only after the caller explicitly opts into normal pickle execution.
    """
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=not trust_checkpoint,
        )
    except Exception as exc:
        if trust_checkpoint:
            raise
        raise UnsafeCheckpointError(
            f"Checkpoint '{path}' was rejected by PyTorch's safe loader. "
            "Only use --trust-checkpoint for a legacy file you created or have "
            "independently verified; that option permits arbitrary pickle code execution."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint '{path}' must contain a dictionary payload")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Checkpoint '{path}' is missing a model state dictionary")
    return checkpoint


def load_mlx_checkpoint_meta(path: str) -> dict | None:
    """Load the sibling MLX metadata file, returning None for legacy weights."""
    meta_path = path + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"MLX checkpoint metadata '{meta_path}' must be an object")
    return payload


def mlx_checkpoint_config_payload(
    path: str,
    *,
    allow_legacy_config: bool,
) -> dict | None:
    """Return the full saved config or reject semantically ambiguous weights."""
    meta = load_mlx_checkpoint_meta(path)
    payload = meta.get("config") if meta else None
    if isinstance(payload, dict):
        return payload
    if allow_legacy_config:
        return None
    raise ValueError(
        f"MLX checkpoint '{path}' predates full configuration metadata. "
        "Refusing to guess semantic model fields. Pass --allow-legacy-config "
        "only after verifying the checkpoint's original configuration."
    )


def config_from_checkpoint_payload(
    payload: Any,
    *,
    source: str,
    schema_version: Any = None,
) -> SpakieConfig:
    """Restore a configuration from a new primitive checkpoint payload.

    Old trusted Torch checkpoints may contain a ``SpakieConfig`` instance;
    accepting that object is safe only because reaching here required the
    caller to opt into unsafe pickle loading explicitly.
    """
    if isinstance(payload, SpakieConfig):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint '{source}' is missing full configuration metadata")
    if schema_version != CHECKPOINT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint '{source}' uses unsupported config schema {schema_version!r}; "
            f"expected {CHECKPOINT_CONFIG_SCHEMA_VERSION}."
        )
    return config_from_dict(payload)


def load_mlx_checkpoint_config(
    path: str,
    *,
    allow_legacy_config: bool = False,
) -> SpakieConfig | None:
    """Restore the semantic config saved beside MLX weights.

    Returning ``None`` is reserved for an explicit legacy opt-in. New files
    must carry both a supported schema marker and a complete config object.
    """
    meta = load_mlx_checkpoint_meta(path)
    payload = mlx_checkpoint_config_payload(
        path, allow_legacy_config=allow_legacy_config
    )
    if payload is None:
        return None
    version = meta.get("config_schema_version") if meta else None
    if version != CHECKPOINT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"MLX checkpoint '{path}' uses unsupported config schema {version!r}; "
            f"expected {CHECKPOINT_CONFIG_SCHEMA_VERSION}."
        )
    return config_from_dict(payload)


def load_mlx_model_weights_strict(model: Any, flat: dict, *, path: str) -> None:
    """Load every MLX model tensor with exact key and shape validation."""
    model_flat = {
        key[len("model.") :]: value
        for key, value in flat.items()
        if key.startswith("model.")
    }
    if not model_flat:
        raise ValueError(f"No 'model.*' tensors found in {path}")
    try:
        model.load_weights(list(model_flat.items()), strict=True)
    except Exception as exc:
        raise ValueError(f"MLX checkpoint '{path}' does not exactly match the model: {exc}") from exc
