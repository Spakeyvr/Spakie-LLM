"""MLX runtime helpers: dtype selection, safetensors I/O, grad utilities."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map, tree_unflatten


MLX_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


_DTYPE_MAP = {
    "fp32": mx.float32,
    "fp16": mx.float16,
    "bf16": mx.bfloat16,
}


@dataclass(frozen=True)
class MLXRuntimeSettings:
    precision: str
    dtype: mx.Dtype

    @property
    def description(self) -> str:
        return f"Backend: mlx | Precision: {self.precision}"


def resolve_mlx_runtime(precision_name: str = "auto") -> MLXRuntimeSettings:
    requested = (precision_name or "auto").lower()
    if requested not in MLX_PRECISION_CHOICES:
        raise ValueError(
            f"Unsupported precision '{precision_name}'. Choices: {', '.join(MLX_PRECISION_CHOICES)}"
        )
    if requested == "auto":
        requested = "bf16" if mx.metal.is_available() else "fp32"
    return MLXRuntimeSettings(precision=requested, dtype=_DTYPE_MAP[requested])


def configure_metal_limits(
    max_gb: float | None = None, wired_gb: float | None = None
) -> dict[str, int]:
    """Optionally raise Metal memory/wired limits on Apple Silicon.

    Defaults (None) preserve MLX's built-in behavior. On a 128 GB machine,
    values like max_gb=96, wired_gb=64 let the 300m preset run comfortably.
    Returns a dict of whatever was actually applied (in bytes).
    """
    applied: dict[str, int] = {}
    if not mx.metal.is_available():
        return applied
    if max_gb is not None and max_gb > 0:
        limit = int(max_gb * (1024 ** 3))
        mx.metal.set_memory_limit(limit)
        applied["memory_limit"] = limit
    if wired_gb is not None and wired_gb > 0:
        limit = int(wired_gb * (1024 ** 3))
        mx.metal.set_wired_limit(limit)
        applied["wired_limit"] = limit
    return applied


def flatten_tree(tree: Any) -> dict[str, mx.array]:
    """Flatten a PyTree of arrays into a dotted-path dict of mx.arrays."""
    return dict(tree_flatten(tree))


def unflatten_tree(flat: dict[str, mx.array]) -> Any:
    return tree_unflatten(list(flat.items()))


def save_safetensors(path: str, tree: Any, metadata: dict[str, str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mx.save_safetensors(path, flatten_tree(tree), metadata=metadata or {})


def load_safetensors(path: str) -> dict[str, mx.array]:
    return mx.load(path)


def save_meta_json(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_meta_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def global_grad_norm(grads: Any) -> mx.array:
    """L2 norm across a PyTree of gradient arrays."""
    flat = tree_flatten(grads)
    total = mx.zeros((), dtype=mx.float32)
    for _, g in flat:
        if g is None:
            continue
        total = total + (g.astype(mx.float32) ** 2).sum()
    return mx.sqrt(total)


def clip_grads(grads: Any, max_norm: float) -> tuple[Any, mx.array]:
    """Scale grads so their global L2 norm is <= max_norm. Returns (clipped_grads, pre_clip_norm)."""
    norm = global_grad_norm(grads)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), max_norm / (norm + 1e-6))

    def _scale(x):
        if x is None:
            return None
        return (x.astype(mx.float32) * scale).astype(x.dtype)

    return tree_map(_scale, grads), norm


def add_grad_trees(a: Any, b: Any) -> Any:
    """Sum two grad PyTrees elementwise. Skips Nones."""

    def _add(x, y):
        if x is None:
            return y
        if y is None:
            return x
        return x + y

    return tree_map(_add, a, b)


def scale_grad_tree(grads: Any, factor: float) -> Any:
    def _mul(x):
        if x is None:
            return None
        return x * factor

    return tree_map(_mul, grads)
