"""CLI entry point for SFT fine-tuning."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.default import checkpoint_search_dirs, get_preset_config, inherit_model_shape
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer
from training.dataset import ChatSFTDataset, train_val_split
from training.finetune import finetune


def resolve_named_checkpoint(config, requested: str | None, default_name: str) -> str:
    candidate = requested or default_name
    if os.path.isabs(candidate) or os.path.dirname(candidate):
        return candidate

    for directory in checkpoint_search_dirs(config):
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return os.path.join(config.checkpoint_dir, candidate)


def default_sft_output_name(train_jsonl_path: str) -> str:
    basename = os.path.basename(train_jsonl_path).lower()
    if basename == "train_mixed.jsonl":
        return "sft_mixed_best.pt"
    if basename == "train_targeted.jsonl":
        return "sft_targeted_best.pt"
    return "sft_best.pt"


def interrupt_name_for(best_name: str) -> str:
    if best_name.endswith("_best.pt"):
        return best_name.replace("_best.pt", "_interrupt.pt")
    if best_name.endswith(".pt"):
        return best_name[:-3] + "_interrupt.pt"
    return best_name + "_interrupt.pt"


def main():
    parser = argparse.ArgumentParser(description="CLI entry point for SFT fine-tuning")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use (92m or 180m)")
    parser.add_argument("--train-jsonl", type=str, default="", help="Path to SFT JSONL file")
    parser.add_argument("--source-checkpoint", type=str, default="", help="Checkpoint filename or path to fine-tune from")
    parser.add_argument("--output-name", type=str, default="", help="Filename for the best SFT checkpoint")
    parser.add_argument("--smoke", action="store_true", help="Run a one-epoch subset smoke test")
    parser.add_argument("--max-examples", type=int, default=0, help="Optional cap on loaded SFT examples")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    output_checkpoint_dir = os.path.join(config.checkpoint_dir, "smoke_sft") if args.smoke else config.checkpoint_dir
    if args.smoke:
        config.sft_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")

    default_jsonl = os.path.join(config.chat_data_dir, "train_mixed.jsonl")
    if not os.path.exists(default_jsonl):
        legacy_jsonl = os.path.join(config.chat_data_dir, "train.jsonl")
        default_jsonl = legacy_jsonl if os.path.exists(legacy_jsonl) else default_jsonl
    jsonl_path = args.train_jsonl or default_jsonl

    # Load pretrained checkpoint
    ckpt_path = resolve_named_checkpoint(config, args.source_checkpoint or None, "pretrain_best.pt")
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

    # Data
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

    output_name = args.output_name or default_sft_output_name(jsonl_path)
    finetune(
        model,
        train_ds,
        val_ds,
        config,
        device,
        best_checkpoint_name=output_name,
        interrupt_checkpoint_name=interrupt_name_for(output_name),
    )


if __name__ == "__main__":
    main()
