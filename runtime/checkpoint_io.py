"""Checkpoint metadata and safe PyTorch deserialization helpers."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from configs.default import (
    CHECKPOINT_CONFIG_SCHEMA_VERSION,
    SpakieConfig,
    config_from_dict,
)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be read by PyTorch's restricted loader."""


def atomic_torch_save(payload: dict, path: str) -> None:
    """Durably replace a Torch checkpoint without exposing partial bytes."""
    import torch

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        torch.save(payload, temp_path)
        with open(temp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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


def discard_training_state(checkpoint: dict) -> None:
    """Release state that inference and fresh SFT runs never consume."""
    for key in ("optimizer", "train_sampler", "rng_state", "scaler"):
        checkpoint.pop(key, None)


def load_mlx_checkpoint_meta(path: str) -> dict | None:
    """Load authoritative embedded MLX metadata, with sidecar fallback."""
    from runtime.mlx_backend import load_safetensors_checkpoint_meta

    payload = load_safetensors_checkpoint_meta(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"MLX checkpoint metadata for '{path}' must be an object")
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
