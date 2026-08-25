"""Linearly interpolate two compatible MLX model checkpoints.

The output uses the base checkpoint's full model/config metadata plus an
explicit interpolation record. Publication is crash safe through the shared
checkpoint writer, and Ctrl+C exits without leaving a partial output.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from runtime.mlx_backend import (
    load_safetensors,
    load_safetensors_checkpoint_meta,
    save_safetensors_checkpoint,
)
from runtime.checkpoint_io import load_mlx_checkpoint_config


def interpolate_arrays(
    base: dict[str, mx.array],
    target: dict[str, mx.array],
    alpha: float,
    selected_keys: set[str] | None = None,
) -> dict[str, mx.array]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if set(base) != set(target):
        missing = sorted(set(base) - set(target))
        extra = sorted(set(target) - set(base))
        raise ValueError(f"checkpoint key mismatch: missing={missing[:3]}, extra={extra[:3]}")

    output: dict[str, mx.array] = {}
    for key, base_value in base.items():
        target_value = target[key]
        if base_value.shape != target_value.shape:
            raise ValueError(
                f"shape mismatch for {key}: {base_value.shape} != {target_value.shape}"
            )
        if selected_keys is not None and key not in selected_keys:
            output[key] = base_value
        else:
            blended = (
                base_value.astype(mx.float32)
                + alpha * (target_value.astype(mx.float32) - base_value.astype(mx.float32))
            )
            output[key] = blended.astype(base_value.dtype)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="target weight from 0 to 1")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--block-start",
        type=int,
        default=None,
        help="if set, interpolate only transformer blocks at or above this zero-based index",
    )
    parser.add_argument(
        "--include-final-norm",
        action="store_true",
        help="with --block-start, also interpolate model.ln_f.weight",
    )
    parser.add_argument(
        "--component",
        choices=("all", "attention", "mlp"),
        default="all",
        help="with --block-start, restrict selected block tensors by component",
    )
    parser.add_argument(
        "--include-block-norms",
        action="store_true",
        help="with a restricted component, also interpolate block layer norms",
    )
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if os.path.abspath(args.output) in {os.path.abspath(args.base), os.path.abspath(args.target)}:
        parser.error("--output must not overwrite an input checkpoint")

    # Reject old or incomplete formats before loading multi-gigabyte tensors.
    load_mlx_checkpoint_config(args.base)
    load_mlx_checkpoint_config(args.target)
    base_meta = load_safetensors_checkpoint_meta(args.base)
    target_meta = load_safetensors_checkpoint_meta(args.target)
    if not base_meta or not target_meta:
        raise ValueError("both inputs require checkpoint metadata")
    architecture_keys = (
        "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
        "max_seq_len", "bias", "norm_type", "mlp_type", "qk_norm",
        "position_encoding", "rope_theta",
        "residual_type", "swiglu_hidden",
    )
    base_config = base_meta.get("config", {})
    target_config = target_meta.get("config", {})
    architecture_differences = {
        key: {"base": base_config.get(key), "target": target_config.get(key)}
        for key in architecture_keys
        if base_config.get(key) != target_config.get(key)
    }
    if architecture_differences:
        raise ValueError(f"checkpoint architectures differ: {architecture_differences}")
    if base_meta.get("tokenizer") != target_meta.get("tokenizer"):
        raise ValueError("checkpoint tokenizer contracts differ")

    base_arrays = load_safetensors(args.base)
    target_arrays = load_safetensors(args.target)
    selected_keys = None
    if args.block_start is not None:
        if args.block_start < 0:
            parser.error("--block-start must be nonnegative")
        prefix = "model.blocks."
        block_keys = {
            key
            for key in base_arrays
            if key.startswith(prefix)
            and int(key[len(prefix):].split(".", 1)[0]) >= args.block_start
        }
        if args.component == "attention":
            selected_keys = {key for key in block_keys if ".attn." in key}
        elif args.component == "mlp":
            selected_keys = {key for key in block_keys if ".mlp." in key}
        else:
            selected_keys = block_keys
        if args.component != "all" and args.include_block_norms:
            selected_keys.update(
                key for key in block_keys if ".ln1." in key or ".ln2." in key
            )
        if args.include_final_norm:
            selected_keys.add("model.ln_f.weight")
        if not selected_keys:
            parser.error("--block-start selected no checkpoint tensors")
    arrays = interpolate_arrays(base_arrays, target_arrays, args.alpha, selected_keys)
    if arrays:
        mx.eval(*arrays.values())
    meta = dict(base_meta)
    meta.pop("checkpoint_generation", None)
    meta["interpolation"] = {
        "base": args.base,
        "target": args.target,
        "target_alpha": args.alpha,
        "base_alpha": 1.0 - args.alpha,
        "metadata_config_source": "base",
        "selected_scope": (
            "all_tensors"
            if args.block_start is None
            else {
                "transformer_block_start": args.block_start,
                "include_final_norm": args.include_final_norm,
                "component": args.component,
                "include_block_norms": args.include_block_norms,
                "tensor_count": len(selected_keys or ()),
            }
        ),
    }
    meta["val_loss"] = None
    save_safetensors_checkpoint(args.output, arrays, meta)
    print(
        f"Saved {args.output} = {1.0 - args.alpha:.3f} * base + "
        f"{args.alpha:.3f} * target"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no partial checkpoint was published.")
        raise SystemExit(130)
