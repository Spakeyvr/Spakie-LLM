"""Run data preparation, pretraining, and SFT as one terminal command."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SUPPORTED_PRESETS, get_preset_config
from training.muon_core import MUON_ADJUST_LR_CHOICES, OPTIMIZER_CHOICES


BEST_LOSS_RE = re.compile(r"Best val loss(?: so far)?:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _extract_best_loss(text: str, current_best: float | None) -> float | None:
    best_loss = current_best
    for part in re.split(r"[\r\n]+", text):
        match = BEST_LOSS_RE.search(part)
        if match:
            best_loss = float(match.group(1))
    return best_loss


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def processed_data_ready(processed_dir: Path) -> tuple[bool, str]:
    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    missing = [str(path) for path in (train_path, val_path) if not path.exists()]
    if missing:
        return False, f"missing {', '.join(missing)}"

    try:
        train_tokens = int(np.load(train_path, mmap_mode="r").shape[0])
        val_tokens = int(np.load(val_path, mmap_mode="r").shape[0])
    except Exception as exc:
        return False, f"could not read processed arrays: {exc}"

    if train_tokens <= 0 or val_tokens <= 0:
        return False, f"empty processed arrays (train={train_tokens}, val={val_tokens})"
    return True, f"train={train_tokens:,} tokens, val={val_tokens:,} tokens"


def append_if_value(command: list[str], flag: str, value: object) -> None:
    if value not in (None, "", 0, 0.0):
        command.extend([flag, str(value)])


def prepare_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "scripts/prepare_data.py"]
    append_if_value(command, "--target_tokens", args.prepare_target_tokens)
    append_if_value(command, "--target_train_tokens", args.prepare_target_train_tokens)
    append_if_value(command, "--report_path", args.report_path)
    append_if_value(command, "--source_glob", args.source_glob)
    append_if_value(command, "--source_dirs", args.source_dirs)
    append_if_value(command, "--tokenizer_threads", args.tokenizer_threads)
    append_if_value(command, "--tokenize_batch_size", args.tokenize_batch_size)
    append_if_value(command, "--tokenize_batch_chars", args.tokenize_batch_chars)
    if args.no_dedup:
        command.append("--no-dedup")
    return command


def train_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/train.py",
        "--preset",
        args.preset,
        "--backend",
        args.backend,
        "--precision",
        args.precision,
    ]
    append_if_value(command, "--max-steps", args.max_steps)
    append_if_value(command, "--target_tokens", args.target_tokens)
    command.extend(["--optimizer", args.optimizer])
    command.extend(["--muon-adjust-lr-fn", args.muon_adjust_lr_fn])
    append_if_value(command, "--muon-ns-steps", args.muon_ns_steps)
    append_if_value(command, "--muon-momentum", args.muon_momentum)
    command.append("--muon-nesterov" if args.muon_nesterov else "--no-muon-nesterov")
    command.append("--muon-qkv-split" if args.muon_qkv_split else "--no-muon-qkv-split")
    if args.allow_adamw_fallback:
        command.append("--allow-adamw-fallback")
    if args.reset_optimizer:
        command.append("--reset-optimizer")
    if args.smoke:
        command.append("--smoke")
    if args.resume:
        command.append("--resume")
    append_if_value(command, "--resume-from", args.resume_from)

    if args.backend == "torch":
        command.extend(["--device", args.device, "--num-workers", str(args.num_workers)])
    else:
        command.append("--mlx-compile" if args.mlx_compile else "--no-mlx-compile")
        command.append("--mlx-prefetch" if args.mlx_prefetch else "--no-mlx-prefetch")
        append_if_value(command, "--mlx-memory-gb", args.mlx_memory_gb)
        append_if_value(command, "--mlx-wired-gb", args.mlx_wired_gb)
        if args.mlx_profile:
            command.append("--mlx-profile")
    return command


def default_pretrain_checkpoint(args: argparse.Namespace) -> str:
    config = get_preset_config(args.preset)
    ext = ".safetensors" if args.backend == "mlx" else ".pt"
    checkpoint_dir = Path(config.checkpoint_dir)
    if args.smoke:
        checkpoint_dir = checkpoint_dir / "smoke_pretrain"
    return str(checkpoint_dir / f"pretrain_best{ext}")


def sft_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/finetune.py",
        "--preset",
        args.preset,
        "--backend",
        args.backend,
        "--precision",
        args.precision,
        "--no-model-prompt",
        "--source-checkpoint",
        args.sft_source_checkpoint or default_pretrain_checkpoint(args),
    ]
    append_if_value(command, "--train-jsonl", args.train_jsonl)
    append_if_value(command, "--output-name", args.sft_output_name)
    append_if_value(command, "--max-examples", args.max_examples)
    command.extend(["--optimizer", args.optimizer])
    command.extend(["--muon-adjust-lr-fn", args.muon_adjust_lr_fn])
    append_if_value(command, "--muon-ns-steps", args.muon_ns_steps)
    append_if_value(command, "--muon-momentum", args.muon_momentum)
    command.append("--muon-nesterov" if args.muon_nesterov else "--no-muon-nesterov")
    command.append("--muon-qkv-split" if args.muon_qkv_split else "--no-muon-qkv-split")
    if args.allow_adamw_fallback:
        command.append("--allow-adamw-fallback")
    if args.smoke:
        command.append("--smoke")

    if args.backend == "torch":
        command.extend(["--device", args.device, "--num-workers", str(args.num_workers)])
    else:
        command.append("--mlx-compile" if args.mlx_compile else "--no-mlx-compile")
        command.append("--mlx-prefetch" if args.mlx_prefetch else "--no-mlx-prefetch")
        append_if_value(command, "--mlx-memory-gb", args.mlx_memory_gb)
        append_if_value(command, "--mlx-wired-gb", args.mlx_wired_gb)
        if args.mlx_profile:
            command.append("--mlx-profile")
    return command


def run_step(name: str, command: list[str], log_handle) -> tuple[int, float | None]:
    print(f"\n=== {name} ===")
    print("$ " + " ".join(command))
    log_handle.write(f"\n=== {name} ===\n")
    log_handle.write("$ " + " ".join(command) + "\n")
    log_handle.flush()

    best_loss = None
    if os.name == "posix" and sys.stdout.isatty():
        return run_step_pty(command, log_handle, best_loss)

    return run_step_pipe(name, command, log_handle, best_loss)


def run_step_pipe(
    name: str,
    command: list[str],
    log_handle,
    best_loss: float | None,
) -> tuple[int, float | None]:
    process = subprocess.Popen(
        command,
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()
            best_loss = _extract_best_loss(line, best_loss)
        return_code = process.wait()
    except KeyboardInterrupt:
        print(f"\nCtrl+C received. Forwarding interrupt to {name}...")
        process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print(f"{name} did not stop after 5 minutes; terminating it.")
            process.terminate()
            return_code = process.wait()
        return return_code or 130, best_loss

    return return_code, best_loss


def run_step_pty(
    command: list[str],
    log_handle,
    best_loss: float | None,
) -> tuple[int, float | None]:
    master_fd, slave_fd = os.openpty()
    process = subprocess.Popen(
        command,
        cwd=repo_root(),
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            text = chunk.decode(errors="replace")
            log_handle.write(text)
            log_handle.flush()
            best_loss = _extract_best_loss(text, best_loss)
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Forwarding interrupt to child process...")
        process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("Child process did not stop after 5 minutes; terminating it.")
            process.terminate()
            return_code = process.wait()
        return return_code or 130, best_loss
    finally:
        os.close(master_fd)

    return return_code, best_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data if needed, then run pretraining followed by SFT."
    )
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="300m", help="Model preset")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="mlx", help="Training backend")
    parser.add_argument("--max-steps", type=int, default=0, help="Pretraining step cap")
    parser.add_argument("--target_tokens", type=int, default=0, help="Pretraining token budget")
    parser.add_argument("--smoke", action="store_true", help="Use smoke settings for both pretrain and SFT")
    parser.add_argument("--resume", action="store_true", help="Resume pretraining from the interrupt checkpoint")
    parser.add_argument("--resume-from", type=str, default="", help="Resume pretraining from a specific checkpoint")

    parser.add_argument("--force-prepare", action="store_true", help="Re-run prepare_data.py even if train/val arrays exist")
    parser.add_argument("--prepare-target-tokens", type=int, default=0, help="Token budget for prepare_data.py")
    parser.add_argument("--prepare-target-train-tokens", type=int, default=0, help="Train-token target for prepare_data.py")
    parser.add_argument("--report-path", type=str, default="", help="Corpus report path for prepare_data.py")
    parser.add_argument("--source-glob", type=str, default="", help="Input glob for prepare_data.py")
    parser.add_argument("--source-dirs", type=str, default="", help="Comma-separated data/raw subdirs for prepare_data.py")
    parser.add_argument("--no-dedup", action="store_true", help="Disable prepare_data.py document deduplication")
    parser.add_argument("--tokenizer-threads", type=int, default=0, help="Tokenizer threads for prepare_data.py")
    parser.add_argument("--tokenize-batch-size", type=int, default=0, help="Tokenizer batch size for prepare_data.py")
    parser.add_argument("--tokenize-batch-chars", type=int, default=0, help="Tokenizer batch char cap for prepare_data.py")

    parser.add_argument("--skip-sft", action="store_true", help="Stop after pretraining")
    parser.add_argument("--train-jsonl", type=str, default="", help="SFT JSONL path")
    parser.add_argument("--sft-source-checkpoint", type=str, default="", help="Checkpoint to fine-tune from")
    parser.add_argument("--sft-output-name", type=str, default="", help="Best SFT checkpoint filename")
    parser.add_argument("--max-examples", type=int, default=0, help="Cap SFT examples")

    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto", help="Torch device")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto", help="Runtime precision")
    parser.add_argument("--num-workers", type=int, default=2, help="Torch DataLoader workers")
    parser.add_argument(
        "--optimizer",
        choices=OPTIMIZER_CHOICES,
        default="muon",
        help="Optimizer to use. Muon is the required default; AdamW is fallback-only and not recommended.",
    )
    parser.add_argument("--allow-adamw-fallback", action="store_true", help="Allow explicit AdamW fallback")
    parser.add_argument("--reset-optimizer", action="store_true", help="Reset optimizer state on resume mismatch")
    parser.add_argument("--muon-adjust-lr-fn", choices=MUON_ADJUST_LR_CHOICES, default="match_rms_adamw")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-qkv-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlx-compile", action=argparse.BooleanOptionalAction, default=True, help="Use MLX compile")
    parser.add_argument("--mlx-prefetch", action=argparse.BooleanOptionalAction, default=True, help="Use MLX prefetch")
    parser.add_argument("--mlx-memory-gb", type=float, default=0.0, help="MLX Metal memory limit in GB")
    parser.add_argument("--mlx-wired-gb", type=float, default=0.0, help="MLX Metal wired-memory limit in GB")
    parser.add_argument("--mlx-profile", action="store_true", help="Print MLX profile buckets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    log_dir = repo_root() / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{stamp}.log"

    results: list[tuple[str, float | None]] = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        processed_dir = repo_root() / "data" / "processed"
        ready, reason = processed_data_ready(processed_dir)
        print(f"Processed data check: {reason}")
        log_handle.write(f"Processed data check: {reason}\n")

        if args.force_prepare or not ready:
            return_code, loss = run_step("Prepare data", prepare_command(args), log_handle)
            if return_code != 0:
                print(f"\nPipeline stopped during data preparation (exit {return_code}).")
                return return_code
            results.append(("prepare", loss))
        else:
            print("Skipping data preparation because processed train/val arrays are ready.")

        return_code, pretrain_loss = run_step("Pretrain", train_command(args), log_handle)
        if return_code != 0:
            print(f"\nPipeline stopped during pretraining (exit {return_code}).")
            return return_code
        results.append(("pretrain", pretrain_loss))

        if not args.skip_sft:
            return_code, sft_loss = run_step("SFT", sft_command(args), log_handle)
            if return_code != 0:
                print(f"\nPipeline stopped during SFT (exit {return_code}).")
                return return_code
            results.append(("sft", sft_loss))

    elapsed = time.time() - started
    loss_summary = ", ".join(
        f"{name}={loss:.4f}" for name, loss in results if loss is not None
    ) or "no validation loss was reported"
    reported_losses = [loss for _, loss in results if loss is not None]
    best_overall = min(reported_losses) if reported_losses else None

    print("\n=== Pipeline complete ===")
    print(f"Elapsed: {format_duration(elapsed)}")
    print(f"Best val loss: {loss_summary}")
    if best_overall is not None:
        print(f"Best overall val loss: {best_overall:.4f}")
    print(f"Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
