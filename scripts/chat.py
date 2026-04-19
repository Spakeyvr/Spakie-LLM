"""CLI entry point for chat inference (PyTorch or MLX backend)."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import checkpoint_search_dirs, get_preset_config, inherit_model_shape
from runtime import DEVICE_CHOICES, PRECISION_CHOICES


_TORCH_EXTS = (".pt",)
_MLX_EXTS = (".safetensors",)


def list_available_checkpoints(checkpoint_dirs: list[str], backend: str) -> list[str]:
    """Return available checkpoint paths for the given backend."""
    exts = _MLX_EXTS if backend == "mlx" else _TORCH_EXTS
    checkpoint_paths: list[str] = []
    seen_names = set()
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.isdir(checkpoint_dir):
            continue
        for name in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, name)
            if name.endswith(exts) and os.path.isfile(path) and name not in seen_names:
                checkpoint_paths.append(path)
                seen_names.add(name)

    preferred_names = (
        "sft_targeted_best",
        "sft_targeted_interrupt",
        "sft_mixed_best",
        "sft_mixed_interrupt",
        "sft_best",
        "sft_interrupt",
        "pretrain_best",
        "pretrain_interrupt",
    )
    preferred_order = {f"{name}{ext}": i for i, name in enumerate(preferred_names) for ext in exts}
    checkpoint_paths.sort(
        key=lambda path: (
            preferred_order.get(os.path.basename(path), 99),
            os.path.basename(path).lower(),
            path.lower(),
        )
    )
    return checkpoint_paths


def resolve_checkpoint_path(
    checkpoint_dirs: list[str],
    checkpoint_arg: str | None,
    model_arg: str | None,
    interactive: bool,
    backend: str,
) -> str | None:
    if checkpoint_arg:
        if os.path.isabs(checkpoint_arg) or os.path.dirname(checkpoint_arg):
            return checkpoint_arg
        for directory in checkpoint_dirs:
            candidate = os.path.join(directory, checkpoint_arg)
            if os.path.exists(candidate):
                return candidate
        return checkpoint_arg

    available_paths = list_available_checkpoints(checkpoint_dirs, backend)
    if not available_paths:
        return None

    alias_to_path = {
        os.path.splitext(os.path.basename(path))[0].lower(): path
        for path in available_paths
    }

    if model_arg:
        requested = model_arg.lower()
        if requested.isdigit():
            index = int(requested) - 1
            if 0 <= index < len(available_paths):
                return available_paths[index]
        if requested in alias_to_path:
            return alias_to_path[requested]
        raise ValueError(
            f"Unknown model '{model_arg}'. Use --list-models to see available choices."
        )

    if interactive and len(available_paths) > 1 and sys.stdin.isatty():
        print("Available models:")
        for idx, path in enumerate(available_paths, start=1):
            print(f"  {idx}. {os.path.splitext(os.path.basename(path))[0]}")

        while True:
            selection = input("Select model [Enter for default 1]: ").strip()
            if not selection:
                return available_paths[0]
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(available_paths):
                    return available_paths[index]
            selected_path = alias_to_path.get(selection.lower())
            if selected_path is not None:
                return selected_path
            print("Invalid selection. Enter a number or model name from the list above.")

    return available_paths[0]


def run_torch_chat(args, config, ckpt_path):
    import torch
    from model.transformer import SpakieGPT
    from runtime import resolve_runtime_settings
    from tokenizer.train_tokenizer import SpakieTokenizer
    from inference.chat import chat_loop

    runtime = resolve_runtime_settings(args.device, args.precision)
    device = runtime.device
    print(f"Backend: torch")
    print(f"Device: {device.type}")
    print(f"Precision: {runtime.precision}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = inherit_model_shape(config, ckpt["config"])
    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tok_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    tokenizer = SpakieTokenizer(tok_path)

    print(f"JSON mode: {args.json_mode}")
    chat_loop(
        model, tokenizer, config, runtime,
        system_msg=args.system,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        json_mode=args.json_mode,
    )


def run_mlx_chat(args, config, ckpt_path):
    from mlx.utils import tree_unflatten

    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer
    from inference.chat_mlx import chat_loop as chat_loop_mlx

    runtime = resolve_mlx_runtime(args.precision)
    print(f"Backend: mlx")
    print(f"Device: metal (mlx)")
    print(f"Precision: {runtime.precision}")

    print(f"Loading checkpoint: {ckpt_path}")
    flat = load_safetensors(ckpt_path)
    model_flat = {k[len("model."):]: v for k, v in flat.items() if k.startswith("model.")}
    if not model_flat:
        print(f"Error: no 'model.*' tensors found in {ckpt_path}")
        sys.exit(1)

    model = SpakieGPTMLX(config)
    model.update(tree_unflatten(list(model_flat.items())))
    if runtime.dtype is not None:
        model.set_dtype(runtime.dtype)
    model.eval()

    tok_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    tokenizer = SpakieTokenizer(tok_path)

    print(f"JSON mode: {args.json_mode}")
    chat_loop_mlx(
        model, tokenizer, config,
        system_msg=args.system,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        json_mode=args.json_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Spakie Chat")
    parser.add_argument("--preset", type=str, default="92m",
                        help="Model preset to use (92m or 180m)")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx",
                        help="Inference backend")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: most recent in checkpoints/)")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint name or list number from checkpoints/")
    parser.add_argument("--list-models", action="store_true",
                        help="List available checkpoints and exit")
    parser.add_argument("--no-model-prompt", action="store_true",
                        help="Skip the interactive model picker and use the default checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer model (default: tokenizer/spakie.model)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--json_mode", action="store_true", help="Enable JSON output mode")
    parser.add_argument("--system", type=str, default="",
                        help="Optional system message")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    checkpoint_dirs = checkpoint_search_dirs(config)
    available_paths = list_available_checkpoints(checkpoint_dirs, args.backend)
    print(f"Preset: {config.preset_name}")
    if args.list_models:
        if not available_paths:
            print(f"Error: no {args.backend} checkpoint found. Train a model first.")
            sys.exit(1)
        print("Available models:")
        for idx, path in enumerate(available_paths, start=1):
            print(f"{idx}. {os.path.splitext(os.path.basename(path))[0]}  ({path})")
        sys.exit(0)

    try:
        ckpt_path = resolve_checkpoint_path(
            checkpoint_dirs=checkpoint_dirs,
            checkpoint_arg=args.checkpoint,
            model_arg=args.model,
            interactive=not args.no_model_prompt,
            backend=args.backend,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if ckpt_path is None:
        print(f"Error: no {args.backend} checkpoint found. Train a model first.")
        sys.exit(1)

    if args.backend == "mlx":
        run_mlx_chat(args, config, ckpt_path)
    else:
        run_torch_chat(args, config, ckpt_path)


if __name__ == "__main__":
    main()
