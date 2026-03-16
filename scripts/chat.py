"""CLI entry point for chat inference."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.chat import chat_loop


def main():
    parser = argparse.ArgumentParser(description="Spakie Chat")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints/sft_best.pt or pretrain_best.pt)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer model (default: tokenizer/spakie.model)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--json_mode", action="store_true", help="Enable JSON output mode")
    parser.add_argument("--system", type=str, default="",
                        help="Optional system message")
    args = parser.parse_args()

    config = SpakieConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        sft_path = os.path.join(config.checkpoint_dir, "sft_best.pt")
        pretrain_path = os.path.join(config.checkpoint_dir, "pretrain_best.pt")
        if os.path.exists(sft_path):
            ckpt_path = sft_path
        elif os.path.exists(pretrain_path):
            ckpt_path = pretrain_path
        else:
            print("Error: no checkpoint found. Train a model first.")
            sys.exit(1)

    # Load
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpakieGPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tok_path = args.tokenizer or (config.tokenizer_prefix + ".model")
    tokenizer = SpakieTokenizer(tok_path)

    print(f"Device: {device}")
    print(f"JSON mode: {args.json_mode}")

    chat_loop(
        model, tokenizer, config, device,
        system_msg=args.system,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        json_mode=args.json_mode,
    )


if __name__ == "__main__":
    main()
