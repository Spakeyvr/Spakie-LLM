"""Preview or run a small, fair pretraining LR/schedule sweep."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.default import SUPPORTED_PRESETS, get_preset_config


def run_name(schedule: str, learning_rate: float) -> str:
    return f"{schedule}-lr-{learning_rate:.0e}".replace("+", "")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    config = get_preset_config(args.preset)
    tokens_per_step = config.pretrain_tokens_per_step()
    steps = math.ceil(args.target_tokens / tokens_per_step)
    warmup_steps = max(1, round(steps * args.warmup_fraction))
    commands: list[list[str]] = []

    for schedule in args.schedules:
        for learning_rate in args.learning_rates:
            output_dir = Path(args.output_root) / args.preset / run_name(schedule, learning_rate)
            command = [
                sys.executable,
                "scripts/train.py",
                "--preset", args.preset,
                "--backend", args.backend,
                "--precision", args.precision,
                "--target_tokens", str(args.target_tokens),
                "--pretrain-lr", str(learning_rate),
                "--pretrain-warmup-steps", str(warmup_steps),
                "--lr-schedule", schedule,
                "--eval-interval", str(args.eval_interval),
                "--checkpoint-interval", str(args.checkpoint_interval),
                "--output-dir", str(output_dir),
            ]
            if args.backend == "torch":
                command.extend(("--device", args.device))
            commands.append(command)
    return commands


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview six 100M-token LR/schedule pilots; pass --execute to run them sequentially."
    )
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="300m")
    parser.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=(4e-4, 6e-4, 8e-4))
    parser.add_argument("--schedules", choices=("cosine", "trapezoid"), nargs="+", default=("cosine", "trapezoid"))
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--output-root", default="checkpoints/ablations")
    parser.add_argument("--execute", action="store_true", help="Run the displayed commands")
    args = parser.parse_args(argv)
    if args.target_tokens <= 0:
        parser.error("--target-tokens must be positive")
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("--learning-rates must all be positive")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)
    print(f"Pilot matrix: {len(commands)} runs x {args.target_tokens:,} tokens")
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        print("Preview only. Re-run with --execute after the data/tokenizer smoke checks pass.")
        return 0

    occupied = []
    for command in commands:
        output_dir = Path(command[command.index("--output-dir") + 1])
        resolved = output_dir if output_dir.is_absolute() else ROOT / output_dir
        if resolved.exists() and any(resolved.iterdir()):
            occupied.append(str(output_dir))
    if occupied:
        print(
            "Refusing to overwrite non-empty pilot directories: " + ", ".join(occupied),
            file=sys.stderr,
        )
        print("Choose a new --output-root or resume the individual run explicitly.", file=sys.stderr)
        return 2

    try:
        for index, command in enumerate(commands, start=1):
            print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        print("\nAblation sweep interrupted; completed run directories were preserved.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as exc:
        print(f"Ablation sweep stopped after exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
