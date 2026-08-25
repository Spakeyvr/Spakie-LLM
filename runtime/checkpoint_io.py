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
from runtime.processed_data import processed_manifest_sha256, tokenizer_contract


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
) -> dict:
    """Load a checkpoint without executing arbitrary pickle globals."""
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise UnsafeCheckpointError(
            f"Checkpoint '{path}' was rejected by PyTorch's safe loader. "
            "Spakie does not execute unrestricted checkpoint pickle payloads."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint '{path}' must contain a dictionary payload")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Checkpoint '{path}' is missing a model state dictionary")
    return checkpoint


def discard_training_state(checkpoint: dict) -> None:
    """Release state that inference and fresh SFT runs never consume."""
    for key in (
        "optimizer", "train_sampler", "rng_state", "scaler",
        "processed_data_manifest_sha256",
    ):
        checkpoint.pop(key, None)


def checkpoint_tokenizer_contract(config: SpakieConfig) -> dict | None:
    path = config.tokenizer_prefix + ".model"
    if not os.path.exists(path):
        return None
    return tokenizer_contract(path)


def checkpoint_processed_data_fingerprint(config: SpakieConfig) -> str | None:
    try:
        return processed_manifest_sha256(config.processed_data_dir)
    except FileNotFoundError:
        return None


def validate_checkpoint_tokenizer(
    metadata: dict | None,
    tokenizer_path: str,
    *,
    source: str,
) -> None:
    saved = metadata.get("tokenizer") if isinstance(metadata, dict) else None
    if not isinstance(saved, dict):
        raise ValueError(
            f"Checkpoint '{source}' has no tokenizer provenance."
        )
    current = tokenizer_contract(tokenizer_path)
    if current != saved:
        raise ValueError(
            f"Checkpoint '{source}' was created with a different tokenizer contract."
        )


def validate_checkpoint_processed_data(
    metadata: dict | None,
    processed_dir: str,
    *,
    source: str,
) -> None:
    saved = (
        metadata.get("processed_data_manifest_sha256")
        if isinstance(metadata, dict)
        else None
    )
    try:
        current = processed_manifest_sha256(processed_dir)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Processed data for checkpoint '{source}' has no completion manifest."
        ) from exc
    if not isinstance(saved, str) or not saved:
        raise ValueError(
            f"Checkpoint '{source}' has no processed-data generation fingerprint."
        )
    if saved != current:
        raise ValueError(
            f"Checkpoint '{source}' was created from a different processed-data generation."
        )


def load_mlx_checkpoint_meta(path: str) -> dict | None:
    """Load authoritative metadata embedded in an MLX checkpoint."""
    from runtime.mlx_backend import load_safetensors_checkpoint_meta

    payload = load_safetensors_checkpoint_meta(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"MLX checkpoint metadata for '{path}' must be an object")
    return payload


def mlx_checkpoint_config_payload(path: str) -> dict:
    """Return the full saved config or reject semantically ambiguous weights."""
    meta = load_mlx_checkpoint_meta(path)
    payload = meta.get("config") if meta else None
    if isinstance(payload, dict):
        return payload
    raise ValueError(
        f"MLX checkpoint '{path}' has no full configuration metadata."
    )


def config_from_checkpoint_payload(
    payload: Any,
    *,
    source: str,
    schema_version: Any = None,
) -> SpakieConfig:
    """Restore a configuration from a new primitive checkpoint payload.

    Only primitive dictionaries from the current schema are accepted.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint '{source}' is missing full configuration metadata")
    if schema_version != CHECKPOINT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint '{source}' uses unsupported config schema {schema_version!r}; "
            f"required schema is {CHECKPOINT_CONFIG_SCHEMA_VERSION}."
        )
    return config_from_dict(payload)


def load_mlx_checkpoint_config(path: str) -> SpakieConfig:
    """Restore the current semantic config saved beside MLX weights."""
    meta = load_mlx_checkpoint_meta(path)
    payload = mlx_checkpoint_config_payload(path)
    version = meta.get("config_schema_version") if meta else None
    if version != CHECKPOINT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"MLX checkpoint '{path}' uses unsupported config schema {version!r}; "
            f"required schema is {CHECKPOINT_CONFIG_SCHEMA_VERSION}."
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
