"""Transactional publication and validation for processed token arrays."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


PROCESSED_DATA_MANIFEST = "processed_data_manifest.json"
PROCESSED_DATA_SCHEMA_VERSION = 1


def manifest_path(processed_dir: Path) -> Path:
    return processed_dir / PROCESSED_DATA_MANIFEST


def invalidate_processed_data(processed_dir: Path) -> None:
    path = manifest_path(processed_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    # The manifest is the commit marker for the two arrays. Make its removal
    # durable before any replacement starts so a power loss cannot resurrect a
    # marker that describes the previous generation.
    _fsync_directory(processed_dir)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_processed_data_manifest(
    train_path: Path,
    val_path: Path,
    *,
    train_tokens: int,
    val_tokens: int,
    dtype,
) -> Path:
    """Publish the completion marker last, after both arrays are durable."""
    processed_dir = train_path.parent
    if val_path.parent != processed_dir:
        raise ValueError("train and validation arrays must share one directory")
    payload = {
        "schema_version": PROCESSED_DATA_SCHEMA_VERSION,
        "dtype": np.dtype(dtype).str,
        "train": {
            "name": train_path.name,
            "tokens": int(train_tokens),
            "bytes": train_path.stat().st_size,
        },
        "val": {
            "name": val_path.name,
            "tokens": int(val_tokens),
            "bytes": val_path.stat().st_size,
        },
    }
    final_path = manifest_path(processed_dir)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".tmp", dir=processed_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, final_path)
        _fsync_directory(processed_dir)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return final_path


def validate_processed_data(processed_dir: Path) -> tuple[bool, str]:
    path = manifest_path(processed_dir)
    if not path.exists():
        return False, f"missing completion manifest {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PROCESSED_DATA_SCHEMA_VERSION:
            return False, "unsupported processed-data manifest version"
        expected_dtype = np.dtype(payload["dtype"])
        details = []
        for split in ("train", "val"):
            entry = payload[split]
            expected_name = f"{split}.npy"
            if entry.get("name") != expected_name:
                return False, f"{split} manifest entry must name {expected_name}"
            array_path = processed_dir / entry["name"]
            if not array_path.exists():
                return False, f"missing {array_path}"
            if array_path.stat().st_size != int(entry["bytes"]):
                return False, f"{split} array size does not match completion manifest"
            array = np.load(array_path, mmap_mode="r")
            if array.ndim != 1:
                return False, f"{split} array must be one-dimensional"
            if array.dtype != expected_dtype:
                return False, f"{split} dtype does not match completion manifest"
            if int(array.shape[0]) != int(entry["tokens"]):
                return False, f"{split} token count does not match completion manifest"
            if int(array.shape[0]) <= 0:
                return False, f"{split} array is empty"
            details.append(f"{split}={int(array.shape[0]):,} tokens")
    except Exception as exc:
        return False, f"invalid processed-data manifest or arrays: {exc}"
    return True, ", ".join(details)
