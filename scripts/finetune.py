"""CLI entry point for SFT fine-tuning."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer
from training.dataset import ChatSFTDataset, train_val_split
from training.finetune import finetune


def main():
    config = SpakieConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained checkpoint
    ckpt_path = os.path.join(config.checkpoint_dir, "pretrain_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: pretrained checkpoint not found at {ckpt_path}")
        print("Run pretraining first: python scripts/train.py")
        sys.exit(1)

    print(f"Loading pretrained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])

    # Data
    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    jsonl_path = os.path.join(config.chat_data_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: SFT data not found at {jsonl_path}")
        sys.exit(1)

    dataset = ChatSFTDataset(jsonl_path, tokenizer, config.max_seq_len)
    print(f"SFT examples: {len(dataset)}")

    train_ds, val_ds = train_val_split(dataset)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    finetune(model, train_ds, val_ds, config, device)


if __name__ == "__main__":
    main()
