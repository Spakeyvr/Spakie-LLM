"""CLI entry point for SFT fine-tuning (PyTorch or MLX backend)."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import checkpoint_search_dirs, get_preset_config, inherit_model_shape
from runtime import DEVICE_CHOICES, PRECISION_CHOICES


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
    basename = os.path.basename(train_jsonl_path).lower()
    if basename == "train_mixed.jsonl":
        return f"sft_mixed_best{ext}"
    if basename == "train_targeted.jsonl":
        return f"sft_targeted_best{ext}"
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
    )


def run_mlx_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer
    from training.dataset_mlx import ChatSFTDatasetMLX, SubsetView, train_val_split_mlx
    from training.finetune_mlx import finetune_mlx

    runtime = resolve_mlx_runtime(args.precision)
    print(f"Backend: mlx")
    print(f"Device: metal (mlx)")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")

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
    config.checkpoint_dir = output_checkpoint_dir
    model = SpakieGPTMLX(config)
    model.update(tree_unflatten(list(model_flat.items())))

    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    if not os.path.exists(jsonl_path):
        print(f"Error: SFT data not found at {jsonl_path}")
        sys.exit(1)

    dataset = ChatSFTDatasetMLX(jsonl_path, tokenizer, config.max_seq_len)
    if args.max_examples > 0:
        dataset = SubsetView(dataset, range(min(args.max_examples, len(dataset))))
    elif args.smoke:
        dataset = SubsetView(dataset, range(min(512, len(dataset))))
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
    )


def main():
    parser = argparse.ArgumentParser(description="CLI entry point for SFT fine-tuning")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use (92m or 180m)")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx",
                        help="Training backend")
    parser.add_argument("--train-jsonl", type=str, default="", help="Path to SFT JSONL file")
    parser.add_argument("--source-checkpoint", type=str, default="", help="Checkpoint filename or path to fine-tune from")
    parser.add_argument("--output-name", type=str, default="", help="Filename for the best SFT checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run a one-epoch subset smoke test")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded SFT examples")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (torch backend)")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    output_checkpoint_dir = os.path.join(config.checkpoint_dir, "smoke_sft") if args.smoke else config.checkpoint_dir
    if args.smoke:
        config.sft_epochs = 1

    default_jsonl = os.path.join(config.chat_data_dir, "train_mixed.jsonl")
    if not os.path.exists(default_jsonl):
        legacy_jsonl = os.path.join(config.chat_data_dir, "train.jsonl")
        default_jsonl = legacy_jsonl if os.path.exists(legacy_jsonl) else default_jsonl
    jsonl_path = args.train_jsonl or default_jsonl
    output_name = args.output_name or default_sft_output_name(jsonl_path, args.backend)

    if args.backend == "mlx":
        run_mlx_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir)
    else:
        run_torch_finetune(args, config, jsonl_path, output_name, output_checkpoint_dir)


if __name__ == "__main__":
    main()
