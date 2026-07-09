"""CLI entry point for chat inference (PyTorch or MLX backend)."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import (
    SUPPORTED_PRESETS,
    checkpoint_search_dirs,
    get_preset_config,
    inherit_attention_shape_from_tensors,
    inherit_mlp_shape_from_tensors,
)
from runtime import DEVICE_CHOICES, PRECISION_CHOICES
from runtime.checkpoint_io import (
    config_from_checkpoint_payload,
    discard_training_state,
    load_mlx_checkpoint_config,
    load_mlx_model_weights_strict,
    load_torch_checkpoint,
)


_TORCH_EXTS = (".pt",)
_MLX_EXTS = (".safetensors",)


def infer_checkpoint_mode(ckpt_path: str) -> str:
    """Infer prompt mode from the checkpoint filename."""
    stem = os.path.splitext(os.path.basename(ckpt_path))[0].lower()
    return "continue" if stem.startswith("pretrain_") else "chat"


def apply_mode_defaults(args) -> None:
    """Fill sampling defaults after auto mode has been resolved."""
    if args.mode == "continue":
        args.temperature = 0.8 if args.temperature is None else args.temperature
        args.top_k = 50 if args.top_k is None else args.top_k
        args.top_p = 0.9 if args.top_p is None else args.top_p
        args.repetition_penalty = 1.0 if args.repetition_penalty is None else args.repetition_penalty
    else:
        args.temperature = 0.1 if args.temperature is None else args.temperature
        args.top_k = 1 if args.top_k is None else args.top_k
        args.top_p = 1.0 if args.top_p is None else args.top_p
        args.repetition_penalty = 1.2 if args.repetition_penalty is None else args.repetition_penalty


def list_available_models(backend: str, preset_name: str | None = None) -> list[tuple[str, str]]:
    """Return (preset, checkpoint_path) pairs for runnable models."""
    preset_names = [preset_name] if preset_name else list(SUPPORTED_PRESETS)
    available_models: list[tuple[str, str]] = []
    for current_preset in preset_names:
        config = get_preset_config(current_preset)
        checkpoint_dirs = checkpoint_search_dirs(config)
        available_models.extend(
            (current_preset, path)
            for path in list_available_checkpoints(checkpoint_dirs, backend)
        )
    return available_models


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
    available_models: list[tuple[str, str]],
    checkpoint_arg: str | None,
    model_arg: str | None,
    interactive: bool,
    preset_name: str | None,
) -> tuple[str, str] | None:
    if checkpoint_arg:
        if os.path.isabs(checkpoint_arg) or os.path.dirname(checkpoint_arg):
            if preset_name:
                return preset_name, checkpoint_arg
            for discovered_preset, path in available_models:
                if os.path.abspath(path) == os.path.abspath(checkpoint_arg):
                    return discovered_preset, checkpoint_arg
            return None, checkpoint_arg
        for discovered_preset, path in available_models:
            if os.path.basename(path) == checkpoint_arg:
                return discovered_preset, path
        if preset_name:
            config = get_preset_config(preset_name)
            for directory in checkpoint_search_dirs(config):
                candidate = os.path.join(directory, checkpoint_arg)
                if os.path.exists(candidate):
                    return preset_name, candidate
        return None, checkpoint_arg

    if not available_models:
        return None

    alias_to_model = {
        os.path.splitext(os.path.basename(path))[0].lower(): (discovered_preset, path)
        for discovered_preset, path in available_models
    }

    if model_arg:
        requested = model_arg.lower()
        if requested.isdigit():
            index = int(requested) - 1
            if 0 <= index < len(available_models):
                return available_models[index]
        if requested in alias_to_model:
            return alias_to_model[requested]
        raise ValueError(
            f"Unknown model '{model_arg}'. Use --list-models to see available choices."
        )

    if interactive and len(available_models) > 1 and sys.stdin.isatty():
        print("Available models:")
        for idx, (discovered_preset, path) in enumerate(available_models, start=1):
            model_name = os.path.splitext(os.path.basename(path))[0]
            print(f"  {idx}. [{discovered_preset}] {model_name}")

        while True:
            selection = input("Select model [Enter for default 1]: ").strip()
            if not selection:
                return available_models[0]
            if selection.isdigit():
                index = int(selection) - 1
                if 0 <= index < len(available_models):
                    return available_models[index]
            selected_model = alias_to_model.get(selection.lower())
            if selected_model is not None:
                return selected_model
            print("Invalid selection. Enter a number or model name from the list above.")

    return available_models[0]


def run_torch_chat(args, config, ckpt_path):
    from model.transformer import SpakieGPT
    from runtime import resolve_runtime_settings
    from tokenizer.train_tokenizer import SpakieTokenizer
    from inference.chat import chat_loop, continuation_loop

    runtime = resolve_runtime_settings(args.device, args.precision)
    device = runtime.device
    print(f"Backend: torch")
    print(f"Device: {device.type}")
    print(f"Precision: {runtime.precision}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = load_torch_checkpoint(
        ckpt_path, map_location=device, trust_checkpoint=args.trust_checkpoint
    )
    if "config" in ckpt:
        config = config_from_checkpoint_payload(
            ckpt["config"], source=ckpt_path,
            schema_version=ckpt.get("config_schema_version"),
        )
    elif not args.allow_legacy_config:
        raise ValueError(
            "Checkpoint has no full config. Pass --allow-legacy-config only for a verified legacy file."
        )
    discard_training_state(ckpt)
    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tok_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    tokenizer = SpakieTokenizer(tok_path)

    if args.mode == "continue":
        continuation_loop(
            model, tokenizer, config, runtime,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            show_special_tokens=args.show_special_tokens,
        )
    else:
        print(f"JSON mode: {args.json_mode}")
        chat_loop(
            model, tokenizer, config, runtime,
            system_msg=args.system,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            json_mode=args.json_mode,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )


def run_mlx_chat(args, config, ckpt_path):
    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer
    from inference.chat_mlx import chat_loop as chat_loop_mlx, continuation_loop as continuation_loop_mlx

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
    saved_config = load_mlx_checkpoint_config(
        ckpt_path, allow_legacy_config=args.allow_legacy_config
    )
    if saved_config is not None:
        config = saved_config
    else:
        config = inherit_attention_shape_from_tensors(config, model_flat)
        config = inherit_mlp_shape_from_tensors(config, model_flat)

    model = SpakieGPTMLX(config)
    load_mlx_model_weights_strict(model, flat, path=ckpt_path)
    del flat, model_flat
    if runtime.dtype is not None:
        model.set_dtype(runtime.dtype)
    model.eval()

    tok_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    tokenizer = SpakieTokenizer(tok_path)

    if args.mode == "continue":
        continuation_loop_mlx(
            model, tokenizer, config,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            show_special_tokens=args.show_special_tokens,
        )
    else:
        print(f"JSON mode: {args.json_mode}")
        chat_loop_mlx(
            model, tokenizer, config,
            system_msg=args.system,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            json_mode=args.json_mode,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )


def main():
    parser = argparse.ArgumentParser(description="Spakie Chat")
    parser.add_argument("--preset", type=str, default=None,
                        help="Model preset to use (default: search all presets with checkpoints)")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx",
                        help="Inference backend")
    parser.add_argument("--mode", choices=("auto", "chat", "continue"), default="auto",
                        help="Prompt mode: auto uses raw continuation for pretrain_* checkpoints")
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum tokens to generate per response")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Penalty for repeating generated tokens")
    parser.add_argument("--show-special-tokens", action="store_true",
                        help="Show generated special tokens in continuation mode")
    parser.add_argument("--json_mode", action="store_true", help="Enable JSON output mode")
    parser.add_argument("--system", type=str, default="",
                        help="Optional system message to include in chat prompts")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow unsafe Python pickle loading for a trusted legacy Torch checkpoint")
    parser.add_argument("--allow-legacy-config", action="store_true",
                        help="Allow shape guessing for verified legacy checkpoints without full config metadata")
    args = parser.parse_args()

    available_models = list_available_models(args.backend, args.preset)
    if args.list_models:
        if not available_models:
            print(f"Error: no {args.backend} checkpoint found. Train a model first.")
            sys.exit(1)
        print("Available models:")
        for idx, (preset_name, path) in enumerate(available_models, start=1):
            print(f"{idx}. [{preset_name}] {os.path.splitext(os.path.basename(path))[0]}  ({path})")
        sys.exit(0)

    try:
        selected_model = resolve_checkpoint_path(
            available_models=available_models,
            checkpoint_arg=args.checkpoint,
            model_arg=args.model,
            interactive=not args.no_model_prompt,
            preset_name=args.preset,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if selected_model is None:
        print(f"Error: no {args.backend} checkpoint found. Train a model first.")
        sys.exit(1)

    selected_preset, ckpt_path = selected_model
    if selected_preset is None:
        print("Error: could not determine the preset for the selected checkpoint. Pass --preset explicitly.")
        sys.exit(1)

    if args.mode == "auto":
        args.mode = infer_checkpoint_mode(ckpt_path)
    apply_mode_defaults(args)

    config = get_preset_config(selected_preset)
    print(f"Preset: {config.preset_name}")
    print(f"Mode: {args.mode}")

    if args.backend == "mlx":
        run_mlx_chat(args, config, ckpt_path)
    else:
        run_torch_chat(args, config, ckpt_path)


if __name__ == "__main__":
    main()
