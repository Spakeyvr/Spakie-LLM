"""Paired MLX training throughput comparison with thermal snapshots."""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9]+)\s+tok/s")
LEGACY_300M_SHAPE = {
    "n_layers": "24",
    "d_model": "1024",
    "n_heads": "16",
    "n_kv_heads": "0",
    "d_ff": "4096",
}


@dataclass
class RunResult:
    label: str
    throughput: int
    output: str
    thermal_before: str
    thermal_after: str


def thermal_status() -> str:
    try:
        proc = subprocess.run(
            ["pmset", "-g", "therm"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return f"thermal status unavailable: {exc}"
    return " | ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def run_benchmark(label: str, extra_args: list[str]) -> RunResult:
    before = thermal_status()
    cmd = [sys.executable, "scripts/benchmark_mlx_training.py", *extra_args]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    after = thermal_status()
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise RuntimeError(f"{label} benchmark failed with exit code {proc.returncode}")
    match = THROUGHPUT_RE.search(proc.stdout)
    if match is None:
        print(proc.stdout, end="")
        raise RuntimeError(f"{label} benchmark did not report throughput")
    return RunResult(
        label=label,
        throughput=int(match.group(1)),
        output=proc.stdout,
        thermal_before=before,
        thermal_after=after,
    )


def print_run(result: RunResult) -> None:
    print(f"\n[{result.label}] {result.throughput:,} tok/s")
    print(f"  thermal before: {result.thermal_before}")
    print(f"  thermal after:  {result.thermal_after}")
    for line in result.output.splitlines():
        if (
            line.startswith("Task:")
            or line.startswith("Preset:")
            or line.startswith("Shape:")
            or line.startswith("Batch size:")
            or line.startswith("Grad accum:")
            or line.startswith("GELU variant:")
            or line.startswith("Norm type:")
            or line.startswith("Loss layout:")
            or line.startswith("Residual type:")
            or line.startswith("Dropout:")
            or line.startswith("Optimizer:")
            or line.startswith("Avg step:")
            or line.startswith("Throughput:")
            or line.startswith("MLX profile")
        ):
            print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--preset", default="300m")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0, help="Optimized run batch-size override")
    parser.add_argument("--grad-accum", type=int, default=0, help="Optimized run grad-accum override")
    parser.add_argument("--mlx-memory-gb", type=float, default=0.0)
    parser.add_argument("--mlx-wired-gb", type=float, default=0.0)
    parser.add_argument("--compile-accum-step", action="store_true", help="Enable compiled grad accumulation for the optimized run")
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default=None, help="Optimized run optimizer override")
    parser.add_argument("--learning-rate", type=float, default=0.0, help="Optimized run learning-rate override")
    parser.add_argument("--grad-clip", type=float, default=None, help="Optimized run grad-clip override; <=0 disables clipping")
    parser.add_argument("--n-layers", type=int, default=0, help="Optimized run layer-count override")
    parser.add_argument("--d-model", type=int, default=0, help="Optimized run hidden-size override")
    parser.add_argument("--n-heads", type=int, default=0, help="Optimized run attention-head override")
    parser.add_argument("--n-kv-heads", type=int, default=-1, help="Optimized run KV-head override; -1 leaves preset default")
    parser.add_argument("--d-ff", type=int, default=0, help="Optimized run FFN-size override")
    args = parser.parse_args()

    stop_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\nStop requested. Finishing the current benchmark before exiting...")

    signal.signal(signal.SIGINT, handle_sigint)

    data_flag = "--real-data" if args.real_data else "--synthetic"
    common = [
        "--task",
        args.task,
        "--preset",
        args.preset,
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        data_flag,
        "--precision",
        "auto",
        "--compile",
        "--prefetch",
    ]
    if args.mlx_memory_gb > 0:
        common.extend(["--mlx-memory-gb", str(args.mlx_memory_gb)])
    if args.mlx_wired_gb > 0:
        common.extend(["--mlx-wired-gb", str(args.mlx_wired_gb)])
    baseline = [
        *common,
        "--no-async-microbatch-eval",
        "--muon-ns-steps",
        "5",
        "--dropout",
        "0.1",
        "--gelu-variant",
        "exact",
        "--norm-type",
        "layernorm",
        "--loss-layout",
        "flat",
        "--grad-clip",
        "1.0",
    ]
    if args.preset == "300m":
        baseline.extend(
            [
                "--n-layers",
                LEGACY_300M_SHAPE["n_layers"],
                "--d-model",
                LEGACY_300M_SHAPE["d_model"],
                "--n-heads",
                LEGACY_300M_SHAPE["n_heads"],
                "--n-kv-heads",
                LEGACY_300M_SHAPE["n_kv_heads"],
                "--d-ff",
                LEGACY_300M_SHAPE["d_ff"],
            ]
        )
    if args.task == "sft":
        baseline.append("--no-trim-sft-padding")
        baseline.append("--no-length-bucketed-sft")
    optimized = [*common]
    if args.batch_size:
        optimized.extend(["--batch-size", str(args.batch_size)])
    if args.grad_accum:
        optimized.extend(["--grad-accum", str(args.grad_accum)])
    if args.compile_accum_step:
        optimized.append("--compile-accum-step")
    if args.optimizer:
        optimized.extend(["--optimizer", args.optimizer])
    if args.learning_rate > 0:
        optimized.extend(["--learning-rate", str(args.learning_rate)])
    if args.grad_clip is not None:
        optimized.extend(["--grad-clip", str(args.grad_clip)])
    if args.n_layers:
        optimized.extend(["--n-layers", str(args.n_layers)])
    if args.d_model:
        optimized.extend(["--d-model", str(args.d_model)])
    if args.n_heads:
        optimized.extend(["--n-heads", str(args.n_heads)])
    if args.n_kv_heads >= 0:
        optimized.extend(["--n-kv-heads", str(args.n_kv_heads)])
    if args.d_ff:
        optimized.extend(["--d-ff", str(args.d_ff)])

    results: list[RunResult] = []
    try:
        for cycle in range(args.cycles):
            order = [("baseline", baseline), ("optimized", optimized)]
            if cycle % 2 == 1:
                order.reverse()
            for label, cmd_args in order:
                if stop_requested:
                    raise KeyboardInterrupt
                result = run_benchmark(label, cmd_args)
                results.append(result)
                print_run(result)
                if args.cooldown_seconds > 0:
                    time.sleep(args.cooldown_seconds)
            print(f"\nCompleted cycle {cycle + 1}/{args.cycles}")
    except KeyboardInterrupt:
        print("\nComparison interrupted; reporting completed runs.")
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    grouped: dict[str, list[int]] = {}
    for result in results:
        grouped.setdefault(result.label, []).append(result.throughput)
    if {"baseline", "optimized"} <= grouped.keys():
        base = median(grouped["baseline"])
        opt = median(grouped["optimized"])
        speedup = (opt / base - 1.0) * 100.0
        print("\nSummary")
        print(f"  baseline median:  {base:,.0f} tok/s")
        print(f"  optimized median: {opt:,.0f} tok/s")
        print(f"  speedup:          {speedup:.1f}%")


if __name__ == "__main__":
    main()
