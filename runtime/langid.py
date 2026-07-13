"""Shared fastText language identification for download and preparation."""

from __future__ import annotations

import os
import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Callable

import requests

from configs.default import SpakieConfig


_MODEL = None
_LOAD_ATTEMPTED = False
_LAST_FAILURE_AT = 0.0
_LOCK = threading.Lock()
_RETRY_SECONDS = 30.0
_DEFAULT_MODEL_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_checksum_valid(path: Path, config: SpakieConfig) -> bool:
    expected = getattr(config, "langid_model_sha256", _DEFAULT_MODEL_SHA256)
    return not expected or _sha256(path) == expected


def load_langid_model(
    config: SpakieConfig,
    *,
    logger: Callable[[str], None] = print,
):
    """Load lid.176 once, atomically downloading the configured model."""
    import time

    global _MODEL, _LOAD_ATTEMPTED, _LAST_FAILURE_AT
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ATTEMPTED and time.monotonic() - _LAST_FAILURE_AT < _RETRY_SECONDS:
        return None
    with _LOCK:
        if _MODEL is not None:
            return _MODEL
        if _LOAD_ATTEMPTED and time.monotonic() - _LAST_FAILURE_AT < _RETRY_SECONDS:
            return None
        _LOAD_ATTEMPTED = True
        try:
            import fasttext
        except ImportError as exc:
            _LAST_FAILURE_AT = time.monotonic()
            logger(f"  langid unavailable: install fasttext to enable ({exc})")
            return None

        model_path = Path(config.langid_model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger(f"  Downloading fastText lid.176 model to {model_path}")
            temp_name = ""
            try:
                response = requests.get(config.langid_model_url, timeout=120)
                response.raise_for_status()
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{model_path.name}.", suffix=".tmp", dir=model_path.parent
                )
                with os.fdopen(fd, "wb") as handle:
                    handle.write(response.content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, model_path)
            except Exception as exc:
                _LAST_FAILURE_AT = time.monotonic()
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
                logger(f"  langid unavailable: failed to download model ({exc})")
                return None
        if not _model_checksum_valid(model_path, config):
            _LAST_FAILURE_AT = time.monotonic()
            logger(f"  langid unavailable: checksum mismatch for {model_path}")
            return None
        try:
            fasttext.FastText.eprint = lambda *_args, **_kwargs: None
            _MODEL = fasttext.load_model(str(model_path))
        except Exception as exc:
            _LAST_FAILURE_AT = time.monotonic()
            logger(f"  langid unavailable: failed to load model ({exc})")
            return None
        return _MODEL


def is_probably_english(
    text: str,
    config: SpakieConfig | None = None,
    *,
    allow_heuristic_fallback: bool = False,
) -> bool:
    sample = text[:3000]
    if not sample:
        return False
    letters = sum(ch.isalpha() for ch in sample)
    spaces = sample.count(" ")
    if letters < 100 or spaces < 20:
        return False
    config = config or SpakieConfig()
    model = load_langid_model(config)
    if model is None:
        if not allow_heuristic_fallback:
            return False
        ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in sample)
        return ascii_letters / max(letters, 1) >= 0.75
    flat = sample.replace("\n", " ").replace("\r", " ")
    predictions = model.f.predict(flat, 1, 0.0, "strict")
    if not predictions:
        return False
    score, label = predictions[0]
    return label == "__label__en" and float(score) >= config.langid_min_confidence


def reset_langid_cache_for_tests() -> None:
    global _MODEL, _LOAD_ATTEMPTED, _LAST_FAILURE_AT
    with _LOCK:
        _MODEL = None
        _LOAD_ATTEMPTED = False
        _LAST_FAILURE_AT = 0.0
