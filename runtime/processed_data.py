"""Transactional publication and validation for processed token arrays."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


PROCESSED_DATA_MANIFEST = "processed_data_manifest.json"
PROCESSED_DATA_SCHEMA_VERSION = 2
TOKENIZER_SPECIAL_PIECES = (
    "<pad>", "<unk>", "<s>", "</s>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|json|>",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_contract(model_path: str | Path) -> dict:
    """Return the exact tokenizer/vocabulary contract used by token arrays."""
    import sentencepiece as spm

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    processor = spm.SentencePieceProcessor(model_file=str(path))
    return {
        "schema_version": 1,
        "sha256": sha256_file(path),
        "vocab_size": int(processor.get_piece_size()),
        "special_token_ids": {
            piece: int(processor.piece_to_id(piece)) for piece in TOKENIZER_SPECIAL_PIECES
        },
    }


def processed_manifest_sha256(processed_dir: str | Path) -> str:
    path = manifest_path(Path(processed_dir))
    if not path.exists():
        raise FileNotFoundError(f"Missing processed-data manifest: {path}")
    return sha256_file(path)


def read_processed_data_manifest(processed_dir: str | Path) -> dict:
    path = manifest_path(Path(processed_dir))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Processed-data manifest {path} must contain an object")
    return payload


def publish_processed_data_manifest(
    train_path: Path,
    val_path: Path,
    *,
    train_tokens: int,
    val_tokens: int,
    dtype,
    tokenizer: dict | None = None,
    preparation: dict | None = None,
    raw_inputs: dict | None = None,
    max_token_id: int | None = None,
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
            "mtime_ns": train_path.stat().st_mtime_ns,
        },
        "val": {
            "name": val_path.name,
            "tokens": int(val_tokens),
            "bytes": val_path.stat().st_size,
            "mtime_ns": val_path.stat().st_mtime_ns,
        },
        "tokenizer": tokenizer,
        "preparation": preparation,
        "raw_inputs": raw_inputs,
        "max_token_id": None if max_token_id is None else int(max_token_id),
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


def validate_processed_data(
    processed_dir: Path,
    *,
    tokenizer_path: str | Path | None = None,
    preparation: dict | None = None,
    require_provenance: bool = False,
) -> tuple[bool, str]:
    path = manifest_path(processed_dir)
    if not path.exists():
        return False, f"missing completion manifest {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != PROCESSED_DATA_SCHEMA_VERSION:
            return False, "unsupported processed-data manifest version"
        saved_tokenizer = payload.get("tokenizer")
        if tokenizer_path is not None:
            if not isinstance(saved_tokenizer, dict):
                return False, "processed-data manifest has no tokenizer provenance"
            current_tokenizer = tokenizer_contract(tokenizer_path)
            if saved_tokenizer != current_tokenizer:
                return False, "processed data was built with a different tokenizer contract"
        elif require_provenance and not isinstance(saved_tokenizer, dict):
            return False, "processed-data manifest has no tokenizer provenance"
        saved_preparation = payload.get("preparation")
        if preparation is not None and saved_preparation != preparation:
            return False, "processed data was built with a different preparation contract"
        if require_provenance:
            if not isinstance(saved_preparation, dict):
                return False, "processed-data manifest has no preparation provenance"
            if not isinstance(payload.get("raw_inputs"), dict):
                return False, "processed-data manifest has no raw-input provenance"
            if payload.get("max_token_id") is None:
                return False, "processed-data manifest has no token-bound metadata"
        if isinstance(saved_tokenizer, dict) and payload.get("max_token_id") is not None:
            max_token_id = int(payload["max_token_id"])
            if max_token_id < 0 or max_token_id >= int(saved_tokenizer["vocab_size"]):
                return False, "processed-data token IDs exceed the saved tokenizer vocabulary"
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
            if array_path.stat().st_mtime_ns != int(entry.get("mtime_ns", -1)):
                return False, f"{split} array changed after manifest publication"
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
