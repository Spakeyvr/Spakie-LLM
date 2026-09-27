"""Preview, run, or summarize the paired Muon Newton-Schulz pilot.

The pilot is a 2x2 factorial against the current recipe: the DeepSeek-V4
hybrid Newton-Schulz schedule (final ``--polish-steps`` iterations use the
(2, -1.5, 0.5) polish cubic) and orthogonalizing the fused SwiGLU gate/up
projection as two matrices. Every arm of a seed shares model initialization,
data order, schedule, and the fixed validation batches, so arms are compared
pairwise against that seed's baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.default import SUPPORTED_PRESETS, get_preset_config
from training.monitor import STATUS_FILENAME, STATUS_HISTORY_FILENAME

# arm name -> (hybrid polish schedule enabled, split gate_up)
ARMS: dict[str, tuple[bool, bool]] = {
    "baseline": (False, False),
    "paper": (True, True),
    "polish": (True, False),
    "split": (False, True),
}
INTERRUPT_CHECKPOINTS = ("pretrain_interrupt.safetensors", "pretrain_interrupt.pt")
# Written into each run directory at launch: the exact train.py arguments, so a
# directory is only reused, resumed, or compared when the request still matches.
RUN_COMMAND_FILENAME = "pilot_command.json"


def pilot_steps(args: argparse.Namespace) -> tuple[int, int]:
    """Optimizer steps (a multiple of the eval interval, so the last step is evaluated) and tokens."""
    tokens_per_step = get_preset_config(args.preset).pretrain_tokens_per_step()
    steps = math.ceil(args.target_tokens / tokens_per_step)
    steps = math.ceil(steps / args.eval_interval) * args.eval_interval
    return steps, steps * tokens_per_step


def run_dir(args: argparse.Namespace, seed: int, arm: str) -> Path:
    return Path(args.output_root) / args.preset / f"seed{seed}" / arm


def build_runs(args: argparse.Namespace) -> list[tuple[int, str, list[str]]]:
    steps, tokens = pilot_steps(args)
    warmup_steps = max(1, round(steps * args.warmup_fraction))
    runs = []
    for seed in args.seeds:
        for arm in args.arms:
            polish, split_gate_up = ARMS[arm]
            command = [
                sys.executable,
                "scripts/train.py",
                "--preset", args.preset,
                "--backend", args.backend,
                "--precision", args.precision,
                "--seed", str(seed),
                "--optimizer", "muon",
                "--max-steps", str(steps),
                "--target_tokens", str(tokens),
                "--pretrain-warmup-steps", str(warmup_steps),
                "--eval-interval", str(args.eval_interval),
                "--muon-ns-polish-steps", str(args.polish_steps if polish else 0),
                "--muon-split-gate-up" if split_gate_up else "--no-muon-split-gate-up",
                "--output-dir", str(run_dir(args, seed, arm)),
            ]
            if args.backend == "torch":
                command.extend(("--device", args.device))
            runs.append((seed, arm, command))
    return runs


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _train_args(command: list[str]) -> list[str]:
    """The train.py arguments, without the interpreter path."""
    return command[1:]


def write_run_command(directory: Path, command: list[str]) -> None:
    directory = _resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RUN_COMMAND_FILENAME).write_text(json.dumps(_train_args(command)) + "\n", encoding="utf-8")


def run_state(directory: Path, command: list[str]) -> str:
    """Classify a run directory against the command this invocation would run.

    'empty', 'complete', 'running', or 'resumable' for a directory launched with
    the same train.py arguments; 'mismatch' when it was launched with different
    ones; 'occupied' when it holds anything else.
    """
    directory = _resolve(directory)
    if not directory.exists() or not any(directory.iterdir()):
        return "empty"
    try:
        saved = json.loads((directory / RUN_COMMAND_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "occupied"
    if saved != _train_args(command):
        return "mismatch"
    if [path.name for path in directory.iterdir()] == [RUN_COMMAND_FILENAME]:
        return "empty"  # recorded, but train.py never started writing
    status_path = directory / STATUS_FILENAME
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        except (OSError, json.JSONDecodeError):
            status = None
        if status in {"complete", "stopped"}:
            return "complete"
        if status in {"starting", "running"}:
            # Still training, or a crashed process that never wrote a final status.
            return "running"
    if any((directory / name).exists() for name in INTERRUPT_CHECKPOINTS):
        return "resumable"
    return "occupied"


def read_eval_curve(directory: Path, eval_interval: int) -> dict[int, float]:
    """Validation loss at each evaluated step, read from the run's status history."""
    history = _resolve(directory) / STATUS_HISTORY_FILENAME
    curve: dict[int, float] = {}
    if not history.exists():
        return curve
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            point = json.loads(line)
        except json.JSONDecodeError:
            continue
        step, val_loss = point.get("step"), point.get("val_loss")
        # Between evals the status carries the previous val_loss forward; only
        # points written at an eval step belong to the curve.
        if isinstance(step, int) and step > 0 and step % eval_interval == 0 and val_loss is not None:
            curve[step] = float(val_loss)
    return dict(sorted(curve.items()))


def steps_to_reach(curve: dict[int, float], target: float) -> float | None:
    """First (linearly interpolated) step at which ``curve`` reaches ``target``."""
    previous = None
    for step, loss in curve.items():
        if loss <= target:
            if previous is None or previous[1] == loss:
                return float(step)
            prev_step, prev_loss = previous
            return prev_step + (step - prev_step) * (prev_loss - target) / (prev_loss - loss)
        previous = (step, loss)
    return None


COMPARABLE_STATES = {"complete", "running", "resumable"}


def summarize(args: argparse.Namespace) -> dict:
    steps, _ = pilot_steps(args)
    commands = {(seed, arm): command for seed, arm, command in build_runs(args)}
    rows = []
    for seed in args.seeds:
        def curve_for(arm: str) -> tuple[str, dict[int, float]]:
            directory = run_dir(args, seed, arm)
            state = run_state(directory, commands[seed, arm])
            # Runs launched with other settings are never compared.
            curve = read_eval_curve(directory, args.eval_interval) if state in COMPARABLE_STATES else {}
            return state, curve

        _, baseline = curve_for("baseline")
        baseline_final = baseline.get(steps)
        tail_steps = [s for s in baseline if s > steps * 0.75]
        for arm in args.arms:
            directory = run_dir(args, seed, arm)
            state, curve = curve_for(arm)
            final = curve.get(steps)
            row = {
                "seed": seed,
                "arm": arm,
                "state": state,
                "evals": len(curve),
                "final_val_loss": final,
                "delta_final": None,
                "delta_tail_mean": None,
                "token_efficiency": None,
            }
            status_path = _resolve(directory) / STATUS_FILENAME
            if state in COMPARABLE_STATES and status_path.exists():
                row["tok_per_sec"] = json.loads(status_path.read_text(encoding="utf-8")).get("tok_per_sec")
            if final is not None and baseline_final is not None:
                row["delta_final"] = final - baseline_final
                shared = [s for s in tail_steps if s in curve]
                if shared:
                    row["delta_tail_mean"] = sum(curve[s] - baseline[s] for s in shared) / len(shared)
                reached = steps_to_reach(curve, baseline_final)
                # >1 means the arm reached the baseline's final loss with fewer tokens.
                row["token_efficiency"] = steps / reached if reached else None
            rows.append(row)
    return {"preset": args.preset, "steps": steps, "eval_interval": args.eval_interval, "runs": rows}


def print_summary(summary: dict) -> None:
    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print(f"Muon pilot ({summary['preset']}, {summary['steps']} steps); deltas are arm - same-seed baseline")
    print(f"{'seed':>6} {'arm':<9} {'state':<10} {'evals':>5} {'final val':>10} "
          f"{'d final':>9} {'d last25%':>10} {'tok-eff':>8} {'tok/s':>8}")
    for row in summary["runs"]:
        print(f"{row['seed']:>6} {row['arm']:<9} {row['state']:<10} {row['evals']:>5} "
              f"{fmt(row['final_val_loss'], '.4f'):>10} {fmt(row['delta_final'], '+.4f'):>9} "
              f"{fmt(row['delta_tail_mean'], '+.4f'):>10} {fmt(row['token_efficiency'], '.3f'):>8} "
              f"{fmt(row.get('tok_per_sec'), '.0f'):>8}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the paired Muon NS-schedule / gate_up-split pilot; --execute runs it, --summarize compares arms."
    )
    parser.add_argument("--preset", choices=SUPPORTED_PRESETS, default="360m")
    parser.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42,),
                        help="Each seed is one paired block: all arms share its initialization and data order")
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+", default=tuple(ARMS),
                        help="Arms to run, in order (baseline is required for comparisons)")
    parser.add_argument("--polish-steps", type=int, default=2,
                        help="Polish iterations for arms using the hybrid schedule (DeepSeek-V4: 2 of 10)")
    parser.add_argument("--target-tokens", type=int, default=100_000_000,
                        help="Per-run budget, rounded up so the final step is an eval step")
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--output-root", default="checkpoints/muon_pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="Run pending arms; completed arms are skipped and interrupted arms resume")
    mode.add_argument("--summarize", action="store_true", help="Compare finished or in-progress arms")
    parser.add_argument("--summary-json", default="", help="Also write the summary to this JSON path")
    args = parser.parse_args(argv)
    if not all(0 <= seed < 2**32 for seed in args.seeds):
        parser.error("--seeds must be in [0, 2**32)")
    if "baseline" not in args.arms:
        parser.error("--arms must include baseline")
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--arms and --seeds must not repeat")
    if args.target_tokens <= 0 or args.eval_interval <= 0:
        parser.error("--target-tokens and --eval-interval must be positive")
    if not 0 < args.polish_steps <= get_preset_config(args.preset).muon_ns_steps:
        parser.error("--polish-steps must be between 1 and the preset's muon_ns_steps")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        summary = summarize(args)
        print_summary(summary)
        if args.summary_json:
            path = Path(args.summary_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return 0

    runs = build_runs(args)
    steps, tokens = pilot_steps(args)
    states = {(seed, arm): run_state(run_dir(args, seed, arm), command) for seed, arm, command in runs}
    print(f"Muon pilot: {len(runs)} runs x {steps:,} steps ({tokens:,} tokens each)")
    for seed, arm, command in runs:
        print(f"[{states[seed, arm]}] {shlex.join(command)}")
    if not args.execute:
        print("Preview only. Re-run with --execute to train; --summarize compares arms.")
        return 0

    occupied = [
        str(run_dir(args, seed, arm))
        for (seed, arm), state in states.items()
        if state in {"occupied", "running", "mismatch"}
    ]
    if occupied:
        print("Refusing to touch run directories that are running, were launched with different "
              "settings, or hold unrelated files: " + ", ".join(occupied), file=sys.stderr)
        print("Choose a new --output-root for different settings.", file=sys.stderr)
        return 2

    try:
        for index, (seed, arm, command) in enumerate(runs, start=1):
            state = states[seed, arm]
            if state == "complete":
                print(f"\n[{index}/{len(runs)}] seed {seed} {arm}: complete, skipping", flush=True)
                continue
            if state == "resumable":
                command = command + ["--resume"]
            else:
                write_run_command(run_dir(args, seed, arm), command)
            print(f"\n[{index}/{len(runs)}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
    except KeyboardInterrupt:
        print("\nMuon pilot interrupted; re-run with --execute to resume.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as exc:
        print(f"Muon pilot stopped after exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    print_summary(summarize(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
