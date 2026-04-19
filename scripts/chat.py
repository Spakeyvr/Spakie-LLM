"""CLI entry point for chat inference."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.default import checkpoint_search_dirs, get_preset_config, inherit_model_shape
from model.transformer import SpakieGPT
from runtime import DEVICE_CHOICES, PRECISION_CHOICES, resolve_runtime_settings
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.chat import chat_loop


def list_available_checkpoints(checkpoint_dirs: list[str]) -> list[str]:
    """Return available checkpoint paths ordered with chat-tuned models first."""
    checkpoint_paths: list[str] = []
    seen_names = set()
    for checkpoint_dir in checkpoint_dirs:
        if not os.path.isdir(checkpoint_dir):
            continue
        for name in os.listdir(checkpoint_dir):
            path = os.path.join(checkpoint_dir, name)
            if name.endswith(".pt") and os.path.isfile(path) and name not in seen_names:
                checkpoint_paths.append(path)
                seen_names.add(name)

    preferred_order = {
        "sft_targeted_best.pt": 0,
        "sft_targeted_interrupt.pt": 1,
        "sft_mixed_best.pt": 2,
        "sft_mixed_interrupt.pt": 3,
        "sft_best.pt": 4,
        "sft_interrupt.pt": 5,
        "pretrain_best.pt": 6,
        "pretrain_interrupt.pt": 7,
    }
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
) -> str | None:
    """Resolve the checkpoint path from explicit args or an interactive selection."""
    if checkpoint_arg:
        if os.path.isabs(checkpoint_arg) or os.path.dirname(checkpoint_arg):
            return checkpoint_arg
        for directory in checkpoint_dirs:
            candidate = os.path.join(directory, checkpoint_arg)
            if os.path.exists(candidate):
                return candidate
        return checkpoint_arg

    available_paths = list_available_checkpoints(checkpoint_dirs)
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


def main():
    parser = argparse.ArgumentParser(description="Spakie Chat")
    parser.add_argument("--preset", type=str, default="92m",
                        help="Model preset to use (92m or 180m)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints/sft_best.pt or pretrain_best.pt)")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint name or list number from checkpoints/ (for example: sft_best)")
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
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    runtime = resolve_runtime_settings(args.device, args.precision)
    device = runtime.device
    checkpoint_dirs = checkpoint_search_dirs(config)

    available_paths = list_available_checkpoints(checkpoint_dirs)
    print(f"Device: {device.type}")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    if args.list_models:
        if not available_paths:
            print("Error: no checkpoint found. Train a model first.")
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
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if ckpt_path is None:
        print("Error: no checkpoint found. Train a model first.")
        sys.exit(1)

    # Load
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


if __name__ == "__main__":
    main()
