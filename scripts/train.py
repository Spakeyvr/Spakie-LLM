"""CLI entry point for pretraining."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from configs.default import get_preset_config
from model.transformer import SpakieGPT
from model.utils import print_model_summary
from training.dataset import PretrainDataset
from training.pretrain import ResumableBatchSampler


def main():
    parser = argparse.ArgumentParser(description="CLI entry point for pretraining")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use (92m or 180m)")
    parser.add_argument("--smoke", action="store_true", help="Run a short 100-step smoke test")
    parser.add_argument(
        "--target_tokens",
        "--target_train_tokens",
        dest="target_tokens",
        type=int,
        default=0,
        help="Override the pretraining token budget",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Override max training steps")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest interrupt checkpoint")
    parser.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint path")
    args = parser.parse_args()

    config = get_preset_config(args.preset)
    if args.target_tokens > 0:
        config.pretrain_target_tokens = args.target_tokens
        config.refresh_derived_fields()
    if args.max_steps > 0:
        config.pretrain_max_steps = args.max_steps
        if args.target_tokens <= 0:
            config.pretrain_target_tokens = config.pretrain_tokens_per_step() * args.max_steps
    if args.smoke:
        config.checkpoint_dir = os.path.join(config.checkpoint_dir, "smoke_pretrain")
        smoke_token_budget = config.pretrain_tokens_per_step() * 100
        config.pretrain_target_tokens = min(config.pretrain_target_tokens or smoke_token_budget, smoke_token_budget)
        config.pretrain_max_steps = min(config.pretrain_max_steps or 100, 100)
        config.pretrain_eval_interval = min(config.pretrain_eval_interval, 50)
        config.pretrain_eval_batches = min(config.pretrain_eval_batches, 4)

    resume_path = args.resume_from
    if args.resume and not resume_path:
        resume_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.pt")

    resume_state = None
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"Error: resume checkpoint not found at {resume_path}")
            sys.exit(1)
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_config = resume_state.get("config")
        if checkpoint_config is not None:
            config = checkpoint_config
        if args.target_tokens > 0:
            config.pretrain_target_tokens = args.target_tokens
            config.refresh_derived_fields()
        if args.max_steps > 0:
            config.pretrain_max_steps = args.max_steps
            if args.target_tokens <= 0:
                config.pretrain_target_tokens = max(
                    config.pretrain_target_tokens,
                    config.pretrain_tokens_per_step() * args.max_steps,
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Tokens/step: {config.pretrain_tokens_per_step():,}")
    print(f"Target train tokens: {config.pretrain_target_tokens:,}")
    print(f"Max steps: {config.pretrain_max_steps:,}")

    # Model
    model = SpakieGPT(config)
    print_model_summary(model)

    # Data
    train_ds = PretrainDataset(os.path.join(config.processed_data_dir, "train.npy"), config.max_seq_len)
    val_ds = PretrainDataset(os.path.join(config.processed_data_dir, "val.npy"), config.max_seq_len)
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences:   {len(val_ds):,}")

    sampler_state = resume_state.get("train_sampler") if resume_state else None
    if sampler_state:
        train_sampler = ResumableBatchSampler.from_state_dict(sampler_state)
    else:
        train_sampler = ResumableBatchSampler(len(train_ds), config.pretrain_batch_size, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.pretrain_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Train
    from training.pretrain import pretrain
    if resume_state:
        print(f"Resuming from: {resume_path}")
        print(f"Resume step: {resume_state.get('step', 0):,}")
        print(f"Resume tokens: {resume_state.get('tokens_processed', 0):,}")
    pretrain(model, train_loader, val_loader, config, device, resume_state=resume_state)


if __name__ == "__main__":
    main()
