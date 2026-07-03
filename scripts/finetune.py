"""CLI entry point for SFT fine-tuning (PyTorch or MLX backend)."""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import (
    SUPPORTED_PRESETS,
    checkpoint_search_dirs,
    get_preset_config,
    inherit_attention_shape_from_tensors,
    inherit_mlp_shape_from_tensors,
    inherit_model_shape,
)
from runtime import DEVICE_CHOICES, PRECISION_CHOICES
from training.muon_core import (
    MUON_ADJUST_LR_CHOICES,
    OPTIMIZER_CHOICES,
    adamw_fallback_warning,
)


_TORCH_EXTS = (".pt",)
_MLX_EXTS = (".safetensors",)


def apply_optimizer_args(config, args) -> None:
    # CLI optimizer flags default to None so per-preset config values stay
    # authoritative unless the user passes an explicit override.
    overrides = {
        "sft_optimizer": args.optimizer,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_momentum": args.muon_momentum,
        "muon_nesterov": args.muon_nesterov,
        "muon_qkv_split": args.muon_qkv_split,
        "grouped_muon": args.grouped_muon,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(config, field_name, value)
    if args.allow_adamw_fallback:
        config.allow_adamw_fallback = True


def print_optimizer_banner(kind: str, *, stage: str) -> None:
    if kind == "adamw":
        print(adamw_fallback_warning(stage))
    else:
        print("Optimizer: Muon (required default)")


def list_available_checkpoints(checkpoint_dirs: list[str], backend: str) -> list[str]:
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
        "pretrain_best",
        "pretrain_interrupt",
        "sft_best",
        "sft_interrupt",
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


def list_available_models(backend: str, preset_name: str | None = None) -> list[tuple[str, str]]:
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


def infer_preset_from_checkpoint(checkpoint_path: str) -> str | None:
    """Best-effort read of `preset_name` from a checkpoint's own metadata.

    MLX checkpoints write `<path>.meta.json`; torch `.pt` files carry a pickled
    `SpakieConfig` (with `preset_name`) inside the archive. Returns None when
    the preset can't be determined or isn't a supported preset.
    """
    if checkpoint_path.endswith(_TORCH_EXTS):
        import torch

        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError):
            return None
        checkpoint_config = ckpt.get("config") if isinstance(ckpt, dict) else None
        preset = getattr(checkpoint_config, "preset_name", None)
        if preset in SUPPORTED_PRESETS:
            return preset
        return None

    meta_path = checkpoint_path + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                preset = json.load(handle).get("preset_name")
        except (OSError, json.JSONDecodeError):
            preset = None
        if preset in SUPPORTED_PRESETS:
            return preset
    return None


def resolve_source_checkpoint(
    available_models: list[tuple[str, str]],
    requested: str | None,
    interactive: bool,
    preset_name: str | None,
    default_name: str,
) -> tuple[str | None, str]:
    candidate = requested or default_name
    if os.path.isabs(candidate) or os.path.dirname(candidate):
        if preset_name:
            return preset_name, candidate
        for discovered_preset, path in available_models:
            if os.path.abspath(path) == os.path.abspath(candidate):
                return discovered_preset, candidate
        # An explicit path that exists but lives outside the standard per-preset
        # checkpoint dirs: infer the preset from its own metadata.
        if os.path.exists(candidate):
            inferred = infer_preset_from_checkpoint(candidate)
            if inferred:
                return inferred, candidate
        return None, candidate

    alias_to_model = {
        os.path.splitext(os.path.basename(path))[0].lower(): (discovered_preset, path)
        for discovered_preset, path in available_models
    }

    if requested:
        if candidate.lower() in alias_to_model:
            return alias_to_model[candidate.lower()]
        for discovered_preset, path in available_models:
            if os.path.basename(path) == candidate:
                return discovered_preset, path
        if preset_name:
            config = get_preset_config(preset_name)
            for directory in checkpoint_search_dirs(config):
                path = os.path.join(directory, candidate)
                if os.path.exists(path):
                    return preset_name, path
        return None, candidate

    if not available_models:
        return preset_name, candidate

    if interactive and len(available_models) > 1 and sys.stdin.isatty():
        print("Available source models:")
        for idx, (discovered_preset, path) in enumerate(available_models, start=1):
            model_name = os.path.splitext(os.path.basename(path))[0]
            print(f"  {idx}. [{discovered_preset}] {model_name}")

        while True:
            selection = input("Select source model [Enter for default 1]: ").strip()
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


def resolve_named_checkpoint(config, requested: str | None, default_name: str) -> str:
    candidate = requested or default_name
    if os.path.isabs(candidate) or os.path.dirname(candidate):
        return candidate

    for directory in checkpoint_search_dirs(config):
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return os.path.join(config.checkpoint_dir, candidate)


def default_sft_output_name(train_jsonl_path: str, backend: str) -> str:
    ext = ".safetensors" if backend == "mlx" else ".pt"
    return f"sft_best{ext}"


def interrupt_name_for(best_name: str) -> str:
    if best_name.endswith("_best.pt"):
        return best_name.replace("_best.pt", "_interrupt.pt")
    if best_name.endswith("_best.safetensors"):
        return best_name.replace("_best.safetensors", "_interrupt.safetensors")
    if best_name.endswith(".pt"):
        return best_name[:-3] + "_interrupt.pt"
    if best_name.endswith(".safetensors"):
        return best_name[: -len(".safetensors")] + "_interrupt.safetensors"
    return best_name + "_interrupt"


def _default_source_checkpoint_name(backend: str) -> str:
    return "pretrain_best.safetensors" if backend == "mlx" else "pretrain_best.pt"


def count_system_prompt_examples(examples: list[dict]) -> int:
    count = 0
    for example in examples:
        messages = example.get("messages", [])
        if messages and messages[0].get("role") == "system":
            count += 1
    return count


def print_sft_format_warning(jsonl_path: str, examples: list[dict]) -> None:
    if not examples:
        return
    system_count = count_system_prompt_examples(examples)
    print(f"SFT system prompts: {system_count:,}/{len(examples):,} examples")
    if system_count == 0:
        print(
            "Warning: SFT data has no system turns. Use `scripts/chat.py` without "
            "`--system` for this checkpoint, or rebuild SFT data with "
            "`scripts/prepare_sft.py --system ...` before fine-tuning."
        )
    elif system_count != len(examples):
        print(
            f"Warning: SFT data at {jsonl_path} mixes examples with and without "
            "system turns; this can make chat behavior less stable."
        )


def run_torch_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir):
    import torch
    from model.transformer import SpakieGPT
    from runtime import resolve_runtime_settings
    from tokenizer.train_tokenizer import SpakieTokenizer
    from training.dataset import ChatSFTDataset, train_val_split
    from training.finetune import finetune

    runtime = resolve_runtime_settings(args.device, args.precision)
    device = runtime.device
    print(f"Backend: torch")
    print(f"Device: {device.type}")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"DataLoader workers: {args.num_workers}")
    print_optimizer_banner(config.sft_optimizer, stage="SFT")

    ckpt_path = resolve_named_checkpoint(
        config, args.source_checkpoint or None, _default_source_checkpoint_name("torch")
    )
    if not os.path.exists(ckpt_path):
        print(f"Error: pretrained checkpoint not found at {ckpt_path}")
        print("Run pretraining first: python scripts/train.py")
        sys.exit(1)

    print(f"Loading pretrained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" in ckpt:
        config = inherit_model_shape(config, ckpt["config"])
    config.checkpoint_dir = output_checkpoint_dir
    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])

    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    if not os.path.exists(jsonl_path):
        print(f"Error: SFT data not found at {jsonl_path}")
        sys.exit(1)

    dataset = ChatSFTDataset(jsonl_path, tokenizer, config.max_seq_len)
    print_sft_format_warning(jsonl_path, dataset.examples)
    if args.max_examples > 0:
        max_examples = min(args.max_examples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_examples))
    elif args.smoke:
        dataset = torch.utils.data.Subset(dataset, range(min(512, len(dataset))))
    print(f"SFT examples: {len(dataset)}")

    train_ds, val_ds = train_val_split(dataset)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    finetune(
        model,
        train_ds,
        val_ds,
        config,
        runtime,
        num_workers=args.num_workers,
        best_checkpoint_name=output_name,
        interrupt_checkpoint_name=interrupt_name_for(output_name),
        allow_adamw_fallback=config.allow_adamw_fallback,
    )


def run_mlx_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import configure_metal_limits, load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer
    from training.dataset_mlx import (
        ChatSFTDatasetMLX,
        PretokenizedChatSFTDatasetMLX,
        SubsetView,
        train_val_split_mlx,
    )
    from training.finetune_mlx import finetune_mlx

    if args.attention_backend is not None:
        config.attention_backend = args.attention_backend
    if args.mlp_checkpointing is not None:
        config.mlp_checkpointing = args.mlp_checkpointing
    if args.compact_valid_mlp is not None:
        config.compact_valid_mlp = args.compact_valid_mlp
    if args.compact_valid_projections is not None:
        config.compact_valid_projections = args.compact_valid_projections
    if args.addmm_residual_projections is not None:
        config.addmm_residual_projections = args.addmm_residual_projections
    if args.contiguous_linear_inputs is not None:
        config.contiguous_linear_inputs = args.contiguous_linear_inputs
    if (
        args.attention_backend is not None
        or args.mlp_checkpointing is not None
        or args.compact_valid_mlp is not None
        or args.compact_valid_projections is not None
        or args.addmm_residual_projections is not None
        or args.contiguous_linear_inputs is not None
    ):
        config.refresh_derived_fields()
    runtime = resolve_mlx_runtime(args.precision)
    applied_limits = configure_metal_limits(
        max_gb=args.mlx_memory_gb if args.mlx_memory_gb > 0 else None,
        wired_gb=args.mlx_wired_gb if args.mlx_wired_gb > 0 else None,
        cache_gb=args.mlx_cache_gb if args.mlx_cache_gb >= 0 else None,
    )
    print(f"Backend: mlx")
    print(f"Device: metal (mlx)")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print_optimizer_banner(config.sft_optimizer, stage="SFT")
    print(
        f"Compile: {args.mlx_compile} | Prefetch: {args.mlx_prefetch} | "
        f"Profile: {args.mlx_profile}"
    )
    print(
        "Microbatch eval: "
        f"loss={args.mlx_eval_microbatch_loss} | "
        f"defer_final={args.mlx_defer_final_microbatch_eval}"
    )
    print(
        f"SFT bucket multiple: {args.sft_bucket_multiple} | "
        f"SFT length bucket size: {args.sft_length_bucket_size} | "
        f"SFT sampler: {args.sft_sampler} | "
        f"Pack SFT: {args.sft_pack} | "
        f"Gather SFT loss: {args.sft_gather_loss} | "
        f"Pretokenize SFT: {args.pretokenize_sft} | "
        f"Async step eval: {args.mlx_async_step_eval}"
    )
    print(f"Attention backend: {config.attention_backend}")
    print(f"MLP checkpointing: {config.mlp_checkpointing}")
    print(f"Compact valid MLP: {config.compact_valid_mlp}")
    print(f"Compact valid projections: {config.compact_valid_projections}")
    print(f"Addmm residual projections: {config.addmm_residual_projections}")
    print(f"Contiguous linear inputs: {config.contiguous_linear_inputs}")
    if applied_limits:
        human_limits = ", ".join(
            f"{k}={v / (1024 ** 3):.1f}GB" for k, v in applied_limits.items()
        )
        print(f"Metal limits: {human_limits}")

    ckpt_path = resolve_named_checkpoint(
        config, args.source_checkpoint or None, _default_source_checkpoint_name("mlx")
    )
    if not os.path.exists(ckpt_path):
        print(f"Error: pretrained checkpoint not found at {ckpt_path}")
        print("Run pretraining first: python scripts/train.py --backend mlx")
        sys.exit(1)

    print(f"Loading pretrained checkpoint: {ckpt_path}")
    flat = load_safetensors(ckpt_path)
    model_flat = {k[len("model."):]: v for k, v in flat.items() if k.startswith("model.")}
    if not model_flat:
        print(f"Error: no 'model.*' tensors found in {ckpt_path}")
        sys.exit(1)
    config = inherit_attention_shape_from_tensors(config, model_flat)
    config = inherit_mlp_shape_from_tensors(config, model_flat)
    config.checkpoint_dir = output_checkpoint_dir
    model = SpakieGPTMLX(config)
    model.update(tree_unflatten(list(model_flat.items())))

    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    if not os.path.exists(jsonl_path):
        print(f"Error: SFT data not found at {jsonl_path}")
        sys.exit(1)

    dataset = ChatSFTDatasetMLX(jsonl_path, tokenizer, config.max_seq_len)
    print_sft_format_warning(jsonl_path, dataset.examples)
    if args.max_examples > 0:
        dataset = SubsetView(dataset, range(min(args.max_examples, len(dataset))))
    elif args.smoke:
        dataset = SubsetView(dataset, range(min(512, len(dataset))))
    if args.pretokenize_sft:
        print("Pretokenizing SFT dataset into in-memory NumPy arrays...")
        try:
            dataset = PretokenizedChatSFTDatasetMLX(dataset)
        except KeyboardInterrupt:
            print("\nInterrupted while pretokenizing SFT dataset.")
            sys.exit(130)
    print(f"SFT examples: {len(dataset)}")

    train_idx, val_idx = train_val_split_mlx(dataset)
    train_ds = SubsetView(dataset, train_idx)
    val_ds = SubsetView(dataset, val_idx)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    finetune_mlx(
        model,
        train_ds,
        val_ds,
        config,
        runtime,
        best_checkpoint_name=output_name,
        interrupt_checkpoint_name=interrupt_name_for(output_name),
        use_compile=args.mlx_compile,
        use_prefetch=args.mlx_prefetch,
        profile=args.mlx_profile,
        eval_microbatch_loss=args.mlx_eval_microbatch_loss,
        defer_final_microbatch_eval=args.mlx_defer_final_microbatch_eval,
        sft_pack=args.sft_pack,
        sft_gather_loss=args.sft_gather_loss,
        sft_bucket_multiple=args.sft_bucket_multiple,
        sft_length_bucket_size=args.sft_length_bucket_size,
        sft_sampler=args.sft_sampler,
        async_step_eval=args.mlx_async_step_eval,
        max_steps=args.max_steps,
        loss_log_path=args.loss_log,
        allow_adamw_fallback=config.allow_adamw_fallback,
    )


def main():
    parser = argparse.ArgumentParser(description="CLI entry point for SFT fine-tuning")
    parser.add_argument(
        "--preset",
        type=str,
        choices=SUPPORTED_PRESETS,
        default=None,
        help="Model preset to use (default: infer from selected source checkpoint)",
    )
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx",
                        help="Training backend")
    parser.add_argument("--train-jsonl", type=str, default="", help="Path to SFT JSONL file")
    parser.add_argument("--source-checkpoint", type=str, default="", help="Checkpoint filename or path to fine-tune from")
    parser.add_argument("--list-models", action="store_true",
                        help="List available source checkpoints and exit")
    parser.add_argument("--no-model-prompt", action="store_true",
                        help="Skip the interactive source model picker and use the default checkpoint")
    parser.add_argument("--output-name", type=str, default="", help="Filename for the best SFT checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run a one-epoch subset smoke test")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded SFT examples")
    parser.add_argument("--epochs", type=int, default=0, help="Override number of SFT epochs")
    parser.add_argument("--sft-batch-size", type=int, default=0, help="Override SFT microbatch size")
    parser.add_argument("--sft-grad-accum", type=int, default=0, help="Override SFT gradient accumulation steps")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop MLX SFT after this many optimizer steps (0 = full epochs)",
    )
    parser.add_argument(
        "--loss-log",
        type=str,
        default="",
        help="Optional CSV path for MLX SFT per-optimizer-step training loss",
    )
    parser.add_argument("--lr", type=float, default=0.0, help="Override SFT learning rate")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (torch backend)")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default=None,
        help="Optimizer to use (default: preset config, normally muon). AdamW is fallback-only and not recommended.",
    )
    parser.add_argument(
        "--allow-adamw-fallback",
        action="store_true",
        help="Allow an explicit AdamW fallback if Muon fails. Never falls back silently.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Accepted for CLI symmetry; SFT starts with fresh optimizer state.",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=MUON_ADJUST_LR_CHOICES,
        default=None,
        help="Muon LR adjustment (default: preset config). match_rms_adamw reuses AdamW-tuned LR/WD.",
    )
    parser.add_argument("--muon-ns-steps", type=int, default=None, help="Muon Newton-Schulz iteration count (default: preset config)")
    parser.add_argument("--muon-momentum", type=float, default=None, help="Muon momentum (default: preset config)")
    parser.add_argument(
        "--muon-nesterov",
        dest="muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Muon Nesterov momentum (default: preset config)",
    )
    parser.add_argument(
        "--muon-qkv-split",
        dest="muon_qkv_split",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply Muon Newton-Schulz to fused Q/K/V chunks independently (default: preset config)",
    )
    parser.add_argument(
        "--grouped-muon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Batch same-shape MLX Muon Newton-Schulz updates across layers (default: preset config)",
    )
    parser.add_argument(
        "--mlx-compile",
        dest="mlx_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the microbatch forward+backward in mx.compile (mlx backend)",
    )
    parser.add_argument(
        "--mlx-prefetch",
        dest="mlx_prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage the next batch on a worker thread (mlx backend)",
    )
    parser.add_argument(
        "--mlx-memory-gb",
        dest="mlx_memory_gb",
        type=float,
        default=0.0,
        help="Set the MLX Metal memory limit in GB (0 = leave default)",
    )
    parser.add_argument(
        "--mlx-profile",
        dest="mlx_profile",
        action="store_true",
        help="Collect rolling MLX timing buckets and print them at epoch boundaries",
    )
    parser.add_argument(
        "--mlx-eval-microbatch-loss",
        dest="mlx_eval_microbatch_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the scalar loss in each MLX microbatch materialization",
    )
    parser.add_argument(
        "--mlx-defer-final-microbatch-eval",
        dest="mlx_defer_final_microbatch_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip microbatch async_eval on the final microbatch before optimizer sync",
    )
    parser.add_argument(
        "--sft-bucket-multiple",
        type=int,
        default=128,
        help="Round trimmed SFT sequence length up to this multiple",
    )
    parser.add_argument(
        "--sft-length-bucket-size",
        type=int,
        default=2048,
        help="Number of examples per SFT sortish length bucket",
    )
    parser.add_argument(
        "--sft-sampler",
        choices=("sortish", "sorted", "homogeneous-step-sorted"),
        default="sortish",
        help="SFT length sampler: sortish buckets, globally sorted shuffled batches, or same-shape optimizer-step groups",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "mfa", "mfa-varlen"),
        default=None,
        help="Override MLX attention backend",
    )
    parser.add_argument(
        "--mlp-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint only MLP submodules during MLX training",
    )
    parser.add_argument(
        "--compact-valid-mlp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run MLX SFT MLP submodules only on fixed-shape non-padding rows",
    )
    parser.add_argument(
        "--compact-valid-projections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run MLX SFT QKV/out projections only on fixed-shape non-padding rows",
    )
    parser.add_argument(
        "--addmm-residual-projections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fuse residual adds into MLX attention/MLP output projections with mx.addmm",
    )
    parser.add_argument(
        "--contiguous-linear-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force row-contiguous inputs before large MLX linear projections",
    )
    parser.add_argument(
        "--sft-pack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pack logical SFT batches into fewer block-causal physical rows",
    )
    parser.add_argument(
        "--sft-gather-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute SFT vocabulary loss only on supervised target positions",
    )
    parser.add_argument(
        "--pretokenize-sft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache MLX SFT input/label arrays in memory before fine-tuning",
    )
    parser.add_argument(
        "--mlx-async-step-eval",
        dest="mlx_async_step_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mx.async_eval after each MLX optimizer step",
    )
    parser.add_argument(
        "--mlx-wired-gb",
        dest="mlx_wired_gb",
        type=float,
        default=0.0,
        help="Set the MLX Metal wired-memory limit in GB (0 = leave default)",
    )
    parser.add_argument(
        "--mlx-cache-gb",
        dest="mlx_cache_gb",
        type=float,
        default=-1.0,
        help="Set the MLX Metal cache limit in GB (-1 = leave default, 0 = disable cache)",
    )
    args = parser.parse_args()

    available_models = list_available_models(args.backend, args.preset)
    if args.list_models:
        if not available_models:
            print(f"Error: no {args.backend} checkpoint found. Train a model first.")
            sys.exit(1)
        print("Available source models:")
        for idx, (preset_name, path) in enumerate(available_models, start=1):
            print(f"{idx}. [{preset_name}] {os.path.splitext(os.path.basename(path))[0]}  ({path})")
        sys.exit(0)

    selected_preset, resolved_source_checkpoint = resolve_source_checkpoint(
        available_models=available_models,
        requested=args.source_checkpoint or None,
        interactive=not args.no_model_prompt,
        preset_name=args.preset,
        default_name=_default_source_checkpoint_name(args.backend),
    )
    if selected_preset is None:
        print(f"Error: pretrained checkpoint not found at {resolved_source_checkpoint}")
        print(f"Run pretraining first: python scripts/train.py{' --backend mlx' if args.backend == 'mlx' else ''}")
        sys.exit(1)

    config = get_preset_config(selected_preset)
    apply_optimizer_args(config, args)
    if args.epochs > 0:
        config.sft_epochs = args.epochs
    if args.sft_batch_size > 0:
        config.sft_batch_size = args.sft_batch_size
    if args.sft_grad_accum > 0:
        config.sft_grad_accum_steps = args.sft_grad_accum
    if args.lr > 0:
        config.sft_lr = args.lr
    args.source_checkpoint = resolved_source_checkpoint
    output_checkpoint_dir = os.path.join(config.checkpoint_dir, "smoke_sft") if args.smoke else config.checkpoint_dir
    if args.smoke:
        config.sft_epochs = 1

    default_jsonl = os.path.join(config.chat_data_dir, "train.jsonl")
    jsonl_path = args.train_jsonl or default_jsonl
    output_name = args.output_name or default_sft_output_name(jsonl_path, args.backend)

    if args.backend == "mlx":
        run_mlx_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir)
    else:
        run_torch_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir)


if __name__ == "__main__":
    main()
