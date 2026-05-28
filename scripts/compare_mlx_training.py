"""Paired MLX benchmark runner for thermal-aware A/B comparisons."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


THROUGHPUT_RE = re.compile(r"^Throughput:\s+([0-9]+(?:\.[0-9]+)?)\s+tok/s", re.MULTILINE)
SUPERVISED_THROUGHPUT_RE = re.compile(
    r"^Supervised throughput:\s+([0-9]+(?:\.[0-9]+)?)\s+tok/s", re.MULTILINE
)
STEP_RE = re.compile(r"^Avg step:\s+([0-9]+(?:\.[0-9]+)?)\s+ms", re.MULTILINE)
ITER_RE = re.compile(r"^Iterations/s:\s+([0-9]+(?:\.[0-9]+)?)", re.MULTILINE)
TOKENS_PER_STEP_RE = re.compile(r"^Tokens/step:\s+([0-9]+)", re.MULTILINE)
SUPERVISED_TOKENS_PER_STEP_RE = re.compile(
    r"^Supervised tokens/step:\s+([0-9]+(?:\.[0-9]+)?)", re.MULTILINE
)
THERM_BEFORE_RE = re.compile(r"^Thermal before:\s+(.+)$", re.MULTILINE)
THERM_AFTER_RE = re.compile(r"^Thermal after:\s+(.+)$", re.MULTILINE)


@dataclass
class RunResult:
    label: str
    round_index: int
    throughput: float
    supervised_throughput: float
    avg_step_ms: float
    iter_per_sec: float
    tokens_per_step: int
    supervised_tokens_per_step: float
    thermal_before: str
    thermal_after: str


def _parse_result(label: str, round_index: int, output: str) -> RunResult:
    throughput_match = THROUGHPUT_RE.search(output)
    supervised_throughput_match = SUPERVISED_THROUGHPUT_RE.search(output)
    step_match = STEP_RE.search(output)
    iter_match = ITER_RE.search(output)
    tokens_per_step_match = TOKENS_PER_STEP_RE.search(output)
    supervised_tokens_per_step_match = SUPERVISED_TOKENS_PER_STEP_RE.search(output)
    if (
        throughput_match is None
        or step_match is None
        or iter_match is None
        or tokens_per_step_match is None
    ):
        raise RuntimeError(f"Could not parse benchmark output for {label} round {round_index}")
    supervised_throughput = (
        float(supervised_throughput_match.group(1))
        if supervised_throughput_match
        else float(throughput_match.group(1))
    )
    supervised_tokens_per_step = (
        float(supervised_tokens_per_step_match.group(1))
        if supervised_tokens_per_step_match
        else float(tokens_per_step_match.group(1))
    )
    before = THERM_BEFORE_RE.search(output)
    after = THERM_AFTER_RE.search(output)
    return RunResult(
        label=label,
        round_index=round_index,
        throughput=float(throughput_match.group(1)),
        supervised_throughput=supervised_throughput,
        avg_step_ms=float(step_match.group(1)),
        iter_per_sec=float(iter_match.group(1)),
        tokens_per_step=int(tokens_per_step_match.group(1)),
        supervised_tokens_per_step=supervised_tokens_per_step,
        thermal_before=before.group(1) if before else "missing",
        thermal_after=after.group(1) if after else "missing",
    )


def _run_once(command: list[str], label: str, round_index: int) -> RunResult:
    print(f"\n[{round_index}] {label}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (completed.stdout or "") + (completed.stderr or "")
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} round {round_index} failed with exit {completed.returncode}")
    return _parse_result(label, round_index, output)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired MLX training benchmarks")
    parser.add_argument("--rounds", type=int, default=2, help="Number of A/B pairs to run")
    parser.add_argument(
        "--candidate-args",
        default="",
        help="Extra arguments for the candidate run, shell-quoted as one string",
    )
    parser.add_argument(
        "--baseline-args",
        default="",
        help="Extra arguments for the baseline run, shell-quoted as one string",
    )
    parser.add_argument(
        "--candidate-label",
        default="candidate",
        help="Label for the candidate run",
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline",
        help="Label for the baseline run",
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to benchmark_mlx_training.py after --",
    )
    args = parser.parse_args()

    benchmark_args = list(args.benchmark_args)
    if benchmark_args and benchmark_args[0] == "--":
        benchmark_args = benchmark_args[1:]
    base = [sys.executable, str(Path(__file__).with_name("benchmark_mlx_training.py"))]
    baseline_cmd = base + benchmark_args + shlex.split(args.baseline_args)
    candidate_cmd = base + benchmark_args + shlex.split(args.candidate_args)

    results: list[RunResult] = []
    try:
        for round_index in range(1, args.rounds + 1):
            if round_index % 2:
                order = [
                    (args.baseline_label, baseline_cmd),
                    (args.candidate_label, candidate_cmd),
                ]
            else:
                order = [
                    (args.candidate_label, candidate_cmd),
                    (args.baseline_label, baseline_cmd),
                ]
            for label, command in order:
                results.append(_run_once(command, label, round_index))
    except KeyboardInterrupt:
        print("\nInterrupted; summarizing completed runs.", file=sys.stderr)
    if not results:
        print("No completed runs.")
        return

    labels = [args.baseline_label, args.candidate_label]
    grouped = {label: [r for r in results if r.label == label] for label in labels}
    print("\nSummary")
    for label in labels:
        group = grouped[label]
        throughputs = [r.throughput for r in group]
        supervised_throughputs = [r.supervised_throughput for r in group]
        steps = [r.avg_step_ms for r in group]
        iters = [r.iter_per_sec for r in group]
        tokens_per_step = sorted(set(r.tokens_per_step for r in group))
        supervised_tokens_per_step = sorted(set(round(r.supervised_tokens_per_step, 1) for r in group))
        if not group:
            print(f"{label}: no completed runs")
            continue
        print(
            f"{label}: n={len(group)} mean_throughput={_mean(throughputs):.0f} tok/s "
            f"mean_supervised={_mean(supervised_throughputs):.0f} tok/s "
            f"mean_iter={_mean(iters):.4f} it/s mean_step={_mean(steps):.2f} ms "
            f"tokens_per_step={tokens_per_step} supervised_tokens_per_step={supervised_tokens_per_step}"
        )
    baseline_mean = _mean([r.throughput for r in grouped[args.baseline_label]])
    candidate_mean = _mean([r.throughput for r in grouped[args.candidate_label]])
    if baseline_mean > 0 and candidate_mean > 0:
        speedup = (candidate_mean / baseline_mean - 1.0) * 100.0
        baseline_iter_mean = _mean([r.iter_per_sec for r in grouped[args.baseline_label]])
        candidate_iter_mean = _mean([r.iter_per_sec for r in grouped[args.candidate_label]])
        iter_speedup = (candidate_iter_mean / baseline_iter_mean - 1.0) * 100.0
        print(f"Token throughput speedup: {speedup:+.1f}%")
        print(f"Iteration speedup: {iter_speedup:+.1f}%")
        baseline_supervised_mean = _mean(
            [r.supervised_throughput for r in grouped[args.baseline_label]]
        )
        candidate_supervised_mean = _mean(
            [r.supervised_throughput for r in grouped[args.candidate_label]]
        )
        supervised_speedup = (candidate_supervised_mean / baseline_supervised_mean - 1.0) * 100.0
        print(f"Supervised throughput speedup: {supervised_speedup:+.1f}%")

    print("\nThermal log")
    for result in results:
        print(
            f"{result.round_index} {result.label}: before=[{result.thermal_before}] "
            f"after=[{result.thermal_after}]"
        )


if __name__ == "__main__":
    main()
