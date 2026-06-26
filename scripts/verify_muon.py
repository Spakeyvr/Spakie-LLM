"""Verify Muon Newton-Schulz parity between PyTorch and MLX/Metal."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from training.muon_core import (
    MUON_BF16_MAX_ABS,
    MUON_BF16_MAX_REL,
    muon_parity_tolerance_scale,
    MUON_FP32_MAX_ABS,
    MUON_FP32_MAX_REL,
    MUON_NS_COEFFICIENTS,
)
from training.optimizers import muon_newton_schulz_torch


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, int]


CASES = (
    Case("square", (64, 64)),
    Case("tall", (128, 32)),
    Case("wide", (32, 128)),
    Case("hidden_mlp", (3072, 768)),
    Case("qkv_chunk", (768, 768)),
)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    # Elementwise relative error is not meaningful for entries whose true
    # value is close to zero. Use a fixed floor so max_rel catches scale-level
    # drift without letting tiny entries dominate the parity gate.
    denom = np.maximum(denom, 1e-1)
    return float(np.max(np.abs(a - b) / denom))


def _make_matrix(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape, dtype=np.float32)
    return (data / np.float32(np.sqrt(shape[1]))).astype(np.float32, copy=False)


def _resolve_torch_reference_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS reference requested but unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def _torch_reference(
    matrix: np.ndarray,
    *,
    dtype: str,
    ns_steps: int,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(matrix.astype(np.float32, copy=False)).to(device)
    if dtype == "bf16":
        tensor = tensor.to(torch.bfloat16)
    result = muon_newton_schulz_torch(
        tensor,
        ns_steps=ns_steps,
        ns_coefficients=MUON_NS_COEFFICIENTS,
        eps=1e-7,
    )
    return result.float().cpu().numpy()


def _mlx_result(matrix: np.ndarray, *, dtype: str, ns_steps: int) -> np.ndarray:
    # MLX can abort the interpreter during import when no Metal device is
    # available. Run the MLX side in a child process so this command can fail
    # with the required validation message instead of crashing the caller.
    with tempfile.TemporaryDirectory(prefix="muon_verify_") as tmp:
        input_path = os.path.join(tmp, "input.npy")
        output_path = os.path.join(tmp, "output.npy")
        np.save(input_path, matrix)
        worker = r"""
import os
import sys
import numpy as np

repo = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]
dtype = sys.argv[4]
ns_steps = int(sys.argv[5])
sys.path.insert(0, repo)

import mlx.core as mx
if not mx.metal.is_available():
    raise RuntimeError("MLX/Metal validation required")

from training.muon_core import MUON_NS_COEFFICIENTS
from training.optimizers_mlx import muon_newton_schulz_mlx

array = mx.array(np.load(input_path))
if dtype == "bf16":
    array = array.astype(mx.bfloat16)
result = muon_newton_schulz_mlx(
    array,
    ns_steps=ns_steps,
    ns_coefficients=MUON_NS_COEFFICIENTS,
    eps=1e-7,
)
mx.eval(result)
np.save(output_path, np.asarray(result.astype(mx.float32)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", worker, str(ROOT), input_path, output_path, dtype, str(ns_steps)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("MLX/Metal validation required")
        return np.load(output_path)


def run_muon_parity_check(
    *,
    include_bf16: bool = True,
    ns_steps: int = 5,
    torch_reference_device: str = "auto",
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    torch_device = _resolve_torch_reference_device(torch_reference_device)
    dtypes = ("fp32", "bf16") if include_bf16 else ("fp32",)
    for dtype in dtypes:
        tolerance_scale = muon_parity_tolerance_scale(ns_steps)
        max_abs_threshold = (MUON_BF16_MAX_ABS if dtype == "bf16" else MUON_FP32_MAX_ABS) * tolerance_scale
        max_rel_threshold = (MUON_BF16_MAX_REL if dtype == "bf16" else MUON_FP32_MAX_REL) * tolerance_scale
        for index, case in enumerate(CASES):
            matrix = _make_matrix(case.shape, seed=20260501 + index)
            torch_out = _torch_reference(matrix, dtype=dtype, ns_steps=ns_steps, device=torch_device)
            mlx_out = _mlx_result(matrix, dtype=dtype, ns_steps=ns_steps)
            if torch_out.shape != mlx_out.shape:
                raise AssertionError(f"{dtype}:{case.name} shape mismatch: {torch_out.shape} != {mlx_out.shape}")
            if not np.isfinite(torch_out).all() or not np.isfinite(mlx_out).all():
                raise AssertionError(f"{dtype}:{case.name} produced non-finite output")
            max_abs = float(np.max(np.abs(torch_out - mlx_out)))
            max_rel = _max_rel(torch_out, mlx_out)
            results[f"{dtype}:{case.name}"] = {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "torch_reference_device": str(torch_device),
            }
            if max_abs > max_abs_threshold or max_rel > max_rel_threshold:
                raise AssertionError(
                    f"{dtype}:{case.name} failed Muon parity: "
                    f"max_abs={max_abs:.6g} (limit {max_abs_threshold:.6g}), "
                    f"max_rel={max_rel:.6g} (limit {max_rel_threshold:.6g})"
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Muon Newton-Schulz parity on MLX/Metal")
    parser.add_argument("--no-bf16", action="store_true", help="Only run fp32 parity")
    parser.add_argument("--ns-steps", type=int, default=5, help="Newton-Schulz iteration count")
    parser.add_argument(
        "--torch-reference-device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="PyTorch reference device. auto prefers MPS so the check compares Metal numerics.",
    )
    args = parser.parse_args()

    try:
        results = run_muon_parity_check(
            include_bf16=not args.no_bf16,
            ns_steps=args.ns_steps,
            torch_reference_device=args.torch_reference_device,
        )
    except RuntimeError as exc:
        if "MLX/Metal validation required" in str(exc):
            print("MLX/Metal validation required", file=sys.stderr)
        else:
            print(f"Muon verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as exc:
        print(f"Muon verification failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Muon MLX/PyTorch Newton-Schulz parity passed.")
    for name, metrics in results.items():
        print(
            f"{name}: max_abs={metrics['max_abs']:.6g} "
            f"max_rel={metrics['max_rel']:.6g} "
            f"torch_ref={metrics['torch_reference_device']}"
        )


if __name__ == "__main__":
    main()
