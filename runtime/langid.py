"""Shared fastText language identification for download and preparation."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Callable

import requests

from configs.default import SpakieConfig


_MODEL = None
_LOAD_ATTEMPTED = False
_LOCK = threading.Lock()


def load_langid_model(
    config: SpakieConfig,
    *,
    logger: Callable[[str], None] = print,
):
    """Load lid.176 once, atomically downloading the configured model."""
    global _MODEL, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL
    with _LOCK:
        if _LOAD_ATTEMPTED:
            return _MODEL
        _LOAD_ATTEMPTED = True
        try:
            import fasttext
        except ImportError as exc:
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
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
                logger(f"  langid unavailable: failed to download model ({exc})")
                return None
        try:
            fasttext.FastText.eprint = lambda *_args, **_kwargs: None
            _MODEL = fasttext.load_model(str(model_path))
        except Exception as exc:
            logger(f"  langid unavailable: failed to load model ({exc})")
            return None
        return _MODEL


def is_probably_english(text: str, config: SpakieConfig | None = None) -> bool:
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
        ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in sample)
        return ascii_letters / max(letters, 1) >= 0.75
    flat = sample.replace("\n", " ").replace("\r", " ")
    predictions = model.f.predict(flat, 1, 0.0, "strict")
    if not predictions:
        return False
    score, label = predictions[0]
    return label == "__label__en" and float(score) >= config.langid_min_confidence


def reset_langid_cache_for_tests() -> None:
    global _MODEL, _LOAD_ATTEMPTED
    with _LOCK:
        _MODEL = None
        _LOAD_ATTEMPTED = False
