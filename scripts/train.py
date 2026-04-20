"""CLI entry point for pretraining (PyTorch or MLX backend)."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import get_preset_config
from runtime import DEVICE_CHOICES, PRECISION_CHOICES


def run_torch_pretrain(args, config):
    import torch
    from torch.utils.data import DataLoader

    from model.transformer import SpakieGPT
    from model.utils import print_model_summary
    from runtime import dataloader_kwargs, resolve_runtime_settings
    from training.dataset import PretrainDataset
    from training.pretrain import ResumableBatchSampler, pretrain

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

    runtime = resolve_runtime_settings(args.device, args.precision)
    device = runtime.device
    print(f"Backend: torch")
    print(f"Device: {device.type}")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Tokens/step: {config.pretrain_tokens_per_step():,}")
    print(f"Target train tokens: {config.pretrain_target_tokens:,}")
    print(f"Max steps: {config.pretrain_max_steps:,}")
    print(f"DataLoader workers: {args.num_workers}")

    model = SpakieGPT(config)
    print_model_summary(model)

    train_ds = PretrainDataset(os.path.join(config.processed_data_dir, "train.npy"), config.max_seq_len)
    val_ds = PretrainDataset(os.path.join(config.processed_data_dir, "val.npy"), config.max_seq_len)
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences:   {len(val_ds):,}")

    sampler_state = resume_state.get("train_sampler") if resume_state else None
    if sampler_state:
        train_sampler = ResumableBatchSampler.from_state_dict(sampler_state)
    else:
        train_sampler = ResumableBatchSampler(len(train_ds), config.pretrain_batch_size, drop_last=True)

    loader_options = dataloader_kwargs(runtime, args.num_workers)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_options)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.pretrain_batch_size,
        shuffle=False,
        **loader_options,
    )

    if resume_state:
        print(f"Resuming from: {resume_path}")
        print(f"Resume step: {resume_state.get('step', 0):,}")
        print(f"Resume tokens: {resume_state.get('tokens_processed', 0):,}")
    pretrain(model, train_loader, val_loader, config, runtime, resume_state=resume_state)


def run_mlx_pretrain(args, config):
    from model.transformer_mlx import SpakieGPTMLX
    from model.utils import print_model_summary
    from runtime.mlx_backend import configure_metal_limits, resolve_mlx_runtime
    from training.dataset_mlx import PretrainDatasetMLX, ResumableBatchSamplerMLX
    from training.pretrain_mlx import (
        load_training_checkpoint_mlx,
        pretrain_mlx,
    )

    resume_path = args.resume_from
    if args.resume and not resume_path:
        resume_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.safetensors")

    resume_state = None
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"Error: resume checkpoint not found at {resume_path}")
            sys.exit(1)
        resume_state = load_training_checkpoint_mlx(resume_path)
        # Config fields aren't serialized in the safetensors; preset hyperparameters
        # come from get_preset_config. Honor CLI overrides for steps/tokens.
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

    runtime = resolve_mlx_runtime(args.precision)
    applied_limits = configure_metal_limits(
        max_gb=args.mlx_memory_gb if args.mlx_memory_gb > 0 else None,
        wired_gb=args.mlx_wired_gb if args.mlx_wired_gb > 0 else None,
    )
    print(f"Backend: mlx")
    print(f"Device: metal (mlx)")
    print(f"Precision: {runtime.precision}")
    print(f"Preset: {config.preset_name}")
    print(f"Checkpoint dir: {config.checkpoint_dir}")
    print(f"Tokens/step: {config.pretrain_tokens_per_step():,}")
    print(f"Target train tokens: {config.pretrain_target_tokens:,}")
    print(f"Max steps: {config.pretrain_max_steps:,}")
    print(f"Compile: {args.mlx_compile} | Prefetch: {args.mlx_prefetch}")
    if applied_limits:
        human_limits = ", ".join(
            f"{k}={v / (1024 ** 3):.1f}GB" for k, v in applied_limits.items()
        )
        print(f"Metal limits: {human_limits}")

    model = SpakieGPTMLX(config)

    train_ds = PretrainDatasetMLX(os.path.join(config.processed_data_dir, "train.npy"), config.max_seq_len)
    val_ds = PretrainDatasetMLX(os.path.join(config.processed_data_dir, "val.npy"), config.max_seq_len)
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences:   {len(val_ds):,}")

    sampler_state = resume_state["meta"].get("sampler") if resume_state else None
    if sampler_state:
        # Revive rng_state from its json-safe form if needed.
        from training.pretrain_mlx import _json_restore
        sampler_state = dict(sampler_state)
        sampler_state["rng_state"] = _json_restore(sampler_state["rng_state"])
        train_sampler = ResumableBatchSamplerMLX.from_state_dict(sampler_state)
    else:
        train_sampler = ResumableBatchSamplerMLX(
            len(train_ds), config.pretrain_batch_size, drop_last=True, seed=0
        )

    if resume_state:
        print(f"Resuming from: {resume_path}")
        print(f"Resume step: {resume_state['meta'].get('step', 0):,}")
        print(f"Resume tokens: {resume_state['meta'].get('tokens_processed', 0):,}")

    pretrain_mlx(
        model,
        train_ds,
        val_ds,
        train_sampler,
        config,
        runtime,
        resume_state=resume_state,
        use_compile=args.mlx_compile,
        use_prefetch=args.mlx_prefetch,
    )


def main():
    parser = argparse.ArgumentParser(description="CLI entry point for pretraining")
    parser.add_argument("--preset", type=str, default="92m", help="Model preset to use (92m or 180m)")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx",
                        help="Training backend — mlx (Apple Silicon) or torch (MPS/CUDA/CPU)")
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
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (torch backend)")
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
        "--mlx-wired-gb",
        dest="mlx_wired_gb",
        type=float,
        default=0.0,
        help="Set the MLX Metal wired-memory limit in GB (0 = leave default)",
    )
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

    if args.backend == "mlx":
        run_mlx_pretrain(args, config)
    else:
        run_torch_pretrain(args, config)


if __name__ == "__main__":
    main()
