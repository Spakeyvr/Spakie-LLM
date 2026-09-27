import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_muon_pilot import (
    build_runs,
    main,
    parse_args,
    pilot_steps,
    run_dir,
    run_state,
    steps_to_reach,
    summarize,
)
from training.monitor import STATUS_FILENAME, STATUS_HISTORY_FILENAME


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_run(directory: Path, curve: dict[int, float], status: str = "complete") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for step, loss in curve.items():
        # A between-eval write that carries the previous val_loss forward.
        lines.append(json.dumps({"step": step - 1, "val_loss": loss + 1.0, "train_loss": 9.0}))
        lines.append(json.dumps({"step": step, "val_loss": loss, "train_loss": 9.0}))
    (directory / STATUS_HISTORY_FILENAME).write_text("\n".join(lines) + "\n")
    (directory / STATUS_FILENAME).write_text(json.dumps({"status": status, "tok_per_sec": 7000.0}))


class MuonPilotCommandTests(unittest.TestCase):
    def test_default_matrix_is_paired_factorial_on_one_seed(self):
        args = parse_args([])
        runs = build_runs(args)
        steps, tokens = pilot_steps(args)

        self.assertEqual([arm for _, arm, _ in runs], ["baseline", "paper", "polish", "split"])
        self.assertEqual(steps % args.eval_interval, 0)
        self.assertGreaterEqual(tokens, args.target_tokens)
        settings = {
            arm: (_flag(command, "--muon-ns-polish-steps"), "--muon-split-gate-up" in command)
            for _, arm, command in runs
        }
        self.assertEqual(settings, {
            "baseline": ("0", False),
            "paper": ("2", True),
            "polish": ("2", False),
            "split": ("0", True),
        })
        for _, _, command in runs:
            self.assertEqual(_flag(command, "--seed"), "42")
            self.assertEqual(_flag(command, "--max-steps"), str(steps))
            self.assertEqual(_flag(command, "--optimizer"), "muon")
            self.assertIn("--no-mlx-vmap-accum-step", command)
        self.assertEqual(len({_flag(command, "--output-dir") for _, _, command in runs}), 4)

    def test_arguments_must_keep_a_baseline_and_valid_polish(self):
        for argv in (["--arms", "paper"], ["--polish-steps", "0"], ["--polish-steps", "11"]):
            with self.assertRaises(SystemExit):
                parse_args(argv)


class MuonPilotStateTests(unittest.TestCase):
    def test_run_state_classifies_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(run_state(root / "missing"), "empty")
            _write_run(root / "done", {50: 3.0})
            self.assertEqual(run_state(root / "done"), "complete")
            _write_run(root / "cut", {50: 3.0}, status="interrupted")
            (root / "cut" / "pretrain_interrupt.safetensors").write_text("x")
            self.assertEqual(run_state(root / "cut"), "resumable")
            _write_run(root / "live", {50: 3.0}, status="running")
            self.assertEqual(run_state(root / "live"), "running")
            (root / "junk").mkdir()
            (root / "junk" / "file").write_text("x")
            self.assertEqual(run_state(root / "junk"), "occupied")

    def test_execute_skips_complete_resumes_interrupted_and_refuses_occupied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            argv = ["--output-root", temp_dir, "--arms", "baseline", "paper", "polish"]
            args = parse_args(argv)
            _write_run(run_dir(args, 42, "baseline"), {50: 3.0})
            paper = run_dir(args, 42, "paper")
            _write_run(paper, {50: 3.0}, status="interrupted")
            (paper / "pretrain_interrupt.safetensors").write_text("x")
            with mock.patch("scripts.run_muon_pilot.subprocess.run") as run:
                self.assertEqual(main(argv + ["--execute"]), 0)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn("--resume", commands[0])
            self.assertEqual(_flag(commands[0], "--output-dir"), str(paper))
            self.assertNotIn("--resume", commands[1])

            (run_dir(args, 42, "polish") / "stray").parent.mkdir(parents=True, exist_ok=True)
            (run_dir(args, 42, "polish") / "stray").write_text("x")
            with mock.patch("scripts.run_muon_pilot.subprocess.run") as run:
                self.assertEqual(main(argv + ["--execute"]), 2)
            run.assert_not_called()


class MuonPilotSummaryTests(unittest.TestCase):
    def test_steps_to_reach_interpolates_between_evals(self):
        curve = {100: 4.0, 200: 3.0, 300: 2.5}
        self.assertEqual(steps_to_reach(curve, 3.5), 150.0)
        self.assertEqual(steps_to_reach(curve, 4.5), 100.0)
        self.assertIsNone(steps_to_reach(curve, 2.0))

    def test_summary_pairs_arms_with_their_seed_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = parse_args(["--output-root", temp_dir, "--arms", "baseline", "paper"])
            with mock.patch("scripts.run_muon_pilot.pilot_steps", return_value=(200, 0)):
                _write_run(run_dir(args, 42, "baseline"), {50: 4.0, 100: 3.4, 150: 3.1, 200: 3.0})
                _write_run(run_dir(args, 42, "paper"), {50: 3.9, 100: 3.3, 150: 3.0, 200: 2.9})
                summary = summarize(args)

        rows = {row["arm"]: row for row in summary["runs"]}
        self.assertEqual(rows["baseline"]["delta_final"], 0.0)
        self.assertAlmostEqual(rows["paper"]["delta_final"], -0.1)
        self.assertAlmostEqual(rows["paper"]["delta_tail_mean"], -0.1)
        self.assertAlmostEqual(rows["paper"]["token_efficiency"], 200 / 150)
        self.assertEqual(rows["paper"]["evals"], 4)


if __name__ == "__main__":
    unittest.main()
