"""CLI entry point for pretraining (PyTorch or MLX backend)."""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import get_preset_config
from runtime import DEVICE_CHOICES, PRECISION_CHOICES
from training.muon_core import (
    MUON_ADJUST_LR_CHOICES,
    OPTIMIZER_CHOICES,
    adamw_fallback_warning,
)


def apply_optimizer_args(config, args) -> None:
    config.pretrain_optimizer = args.optimizer
    config.allow_adamw_fallback = args.allow_adamw_fallback
    config.muon_adjust_lr_fn = args.muon_adjust_lr_fn
    config.muon_ns_steps = args.muon_ns_steps
    config.muon_momentum = args.muon_momentum
    config.muon_nesterov = args.muon_nesterov
    config.muon_qkv_split = args.muon_qkv_split


def optimizer_kind_from_resume_state(resume_state, *, backend: str) -> str:
    if not resume_state:
        return ""
    if backend == "mlx":
        return str(resume_state.get("meta", {}).get("optimizer_kind", "adamw"))
    optimizer_state = resume_state.get("optimizer", {})
    if not isinstance(optimizer_state, dict):
        optimizer_state = {}
    return str(resume_state.get("optimizer_kind") or optimizer_state.get("optimizer_kind", "adamw"))


def check_resume_optimizer(resume_state, requested: str, *, backend: str, reset_optimizer: bool) -> None:
    saved = optimizer_kind_from_resume_state(resume_state, backend=backend)
    if not saved or saved == requested:
        return
    if reset_optimizer:
        resume_state.pop("optimizer", None)
        print(f"Resetting optimizer state: checkpoint used {saved}, requested {requested}.")
        return
    print(
        f"Error: checkpoint optimizer is {saved}, but requested optimizer is {requested}. "
        "Use --reset-optimizer to resume model weights with fresh optimizer state."
    )
    sys.exit(1)


def print_optimizer_banner(kind: str, *, stage: str) -> None:
    if kind == "adamw":
        print(adamw_fallback_warning(stage))
    else:
        print("Optimizer: Muon (required default)")


def verify_muon_for_full_mlx_pretrain(args, config) -> None:
    if args.backend != "mlx" or config.pretrain_optimizer != "muon" or args.smoke:
        return
    from scripts.verify_muon import run_muon_parity_check

    print("Running required Muon MLX/PyTorch parity check before full MLX pretraining...")
    try:
        run_muon_parity_check(include_bf16=True, ns_steps=config.muon_ns_steps)
    except RuntimeError as exc:
        if "MLX/Metal validation required" in str(exc):
            print("MLX/Metal validation required", file=sys.stderr)
        else:
            print(f"Muon verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as exc:
        print(f"Muon verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
    config.muon_verified = True
    print("Muon parity check passed.")


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
        if args.reset_best_loss:
            resume_state["best_val_loss"] = float("inf")
        checkpoint_config = resume_state.get("config")
        if checkpoint_config is not None:
            config = checkpoint_config
        apply_optimizer_args(config, args)
        check_resume_optimizer(
            resume_state,
            config.pretrain_optimizer,
            backend="torch",
            reset_optimizer=args.reset_optimizer,
        )
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
        if args.additional_steps > 0:
            resume_step = int(resume_state.get("step", 0))
            config.pretrain_max_steps = resume_step + args.additional_steps
            if args.target_tokens <= 0:
                resume_tokens = int(resume_state.get("tokens_processed", 0))
                config.pretrain_target_tokens = max(
                    config.pretrain_target_tokens,
                    resume_tokens + config.pretrain_tokens_per_step() * args.additional_steps,
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
    print_optimizer_banner(config.pretrain_optimizer, stage="Pretraining")

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
    try:
        pretrain(model, train_loader, val_loader, config, runtime, resume_state=resume_state)
    except Exception as exc:
        if config.pretrain_optimizer == "muon" and args.allow_adamw_fallback:
            print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
            config.pretrain_optimizer = "adamw"
            pretrain(model, train_loader, val_loader, config, runtime, resume_state=None)
        else:
            raise


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
        if args.reset_best_loss:
            resume_state.setdefault("meta", {})["best_val_loss"] = float("inf")
        # Config fields aren't serialized in the safetensors; preset hyperparameters
        # come from get_preset_config. Honor CLI overrides for steps/tokens.
        apply_optimizer_args(config, args)
        check_resume_optimizer(
            resume_state,
            config.pretrain_optimizer,
            backend="mlx",
            reset_optimizer=args.reset_optimizer,
        )
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
        if args.additional_steps > 0:
            resume_step = int(resume_state["meta"].get("step", 0))
            config.pretrain_max_steps = resume_step + args.additional_steps
            if args.target_tokens <= 0:
                resume_tokens = int(resume_state["meta"].get("tokens_processed", 0))
                config.pretrain_target_tokens = max(
                    config.pretrain_target_tokens,
                    resume_tokens + config.pretrain_tokens_per_step() * args.additional_steps,
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
    print_optimizer_banner(config.pretrain_optimizer, stage="Pretraining")
    print(f"Compile: {args.mlx_compile} | Prefetch: {args.mlx_prefetch} | Profile: {args.mlx_profile}")
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
        if sampler_state.get("resume_exact") is False:
            print(
                "Sampler indices were omitted from the checkpoint because the "
                "dataset is too large for one MLX array; resuming with a fresh "
                "deterministic shuffle."
            )
        train_sampler = ResumableBatchSamplerMLX.from_state_dict(sampler_state)
    else:
        train_sampler = ResumableBatchSamplerMLX(
            len(train_ds), config.pretrain_batch_size, drop_last=True, seed=0
        )

    if resume_state:
        print(f"Resuming from: {resume_path}")
        print(f"Resume step: {resume_state['meta'].get('step', 0):,}")
        print(f"Resume tokens: {resume_state['meta'].get('tokens_processed', 0):,}")

    try:
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
            profile=args.mlx_profile,
        )
    except Exception as exc:
        if config.pretrain_optimizer == "muon" and args.allow_adamw_fallback:
            print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
            config.pretrain_optimizer = "adamw"
            pretrain_mlx(
                model,
                train_ds,
                val_ds,
                train_sampler,
                config,
                runtime,
                resume_state=None,
                use_compile=args.mlx_compile,
                use_prefetch=args.mlx_prefetch,
                profile=args.mlx_profile,
            )
        else:
            raise


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
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=0,
        help="When resuming, train this many optimizer steps beyond the checkpoint step",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Write checkpoints to this directory instead of the preset checkpoint directory",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help="Override pretraining validation/checkpoint interval in optimizer steps",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=0,
        help="Override number of validation batches used at each pretraining eval",
    )
    parser.add_argument(
        "--reset-best-loss",
        action="store_true",
        help="When resuming, reset best loss so the first validation in a new output dir saves",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the latest interrupt checkpoint")
    parser.add_argument("--resume-from", type=str, default="", help="Resume from a specific checkpoint path")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="Execution device (torch backend)")
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto", help="Execution precision")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes (torch backend)")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="muon",
        help="Optimizer to use. Muon is the required default; AdamW is fallback-only and not recommended.",
    )
    parser.add_argument(
        "--allow-adamw-fallback",
        action="store_true",
        help="Allow an explicit AdamW fallback if Muon fails. Never falls back silently.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Resume model/training counters with fresh optimizer state when optimizer kind changed.",
    )
    parser.add_argument(
        "--muon-adjust-lr-fn",
        choices=MUON_ADJUST_LR_CHOICES,
        default="match_rms_adamw",
        help="Muon LR adjustment. match_rms_adamw reuses AdamW-tuned LR/WD.",
    )
    parser.add_argument("--muon-ns-steps", type=int, default=5, help="Muon Newton-Schulz iteration count")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Muon momentum")
    parser.add_argument(
        "--muon-nesterov",
        dest="muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Muon Nesterov momentum",
    )
    parser.add_argument(
        "--muon-qkv-split",
        dest="muon_qkv_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply Muon Newton-Schulz to fused Q/K/V chunks independently",
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
        help="Collect rolling MLX timing buckets and print them at eval boundaries",
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
    apply_optimizer_args(config, args)
    if args.output_dir:
        config.checkpoint_dir = args.output_dir
    if args.eval_interval > 0:
        config.pretrain_eval_interval = args.eval_interval
    if args.eval_batches > 0:
        config.pretrain_eval_batches = args.eval_batches
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

    verify_muon_for_full_mlx_pretrain(args, config)

    if args.backend == "mlx":
        run_mlx_pretrain(args, config)
    else:
        run_torch_pretrain(args, config)


if __name__ == "__main__":
    main()
