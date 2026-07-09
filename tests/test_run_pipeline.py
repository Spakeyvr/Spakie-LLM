from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import scripts.run_pipeline as run_pipeline


_OPTIMIZER_FLAGS = {
    "--optimizer",
    "--muon-adjust-lr-fn",
    "--muon-ns-steps",
    "--muon-momentum",
    "--muon-nesterov",
    "--no-muon-nesterov",
    "--muon-qkv-split",
    "--no-muon-qkv-split",
}


def parse_args(*arguments: str):
    with patch.object(sys, "argv", ["run_pipeline.py", *arguments]):
        return run_pipeline.parse_args()


class PipelineOptimizerForwardingTests(unittest.TestCase):
    def assert_no_optimizer_overrides(self, command: list[str]) -> None:
        self.assertTrue(_OPTIMIZER_FLAGS.isdisjoint(command), command)

    def test_resume_omits_unspecified_optimizer_and_muon_overrides(self):
        args = parse_args("--resume")

        pretrain_command = run_pipeline.train_command(args)
        sft_command = run_pipeline.sft_command(args)

        self.assertIn("--resume", pretrain_command)
        self.assert_no_optimizer_overrides(pretrain_command)
        self.assert_no_optimizer_overrides(sft_command)

    def test_fresh_pipeline_also_leaves_preset_optimizer_authoritative(self):
        args = parse_args()

        self.assert_no_optimizer_overrides(run_pipeline.train_command(args))
        self.assert_no_optimizer_overrides(run_pipeline.sft_command(args))

    def test_explicit_optimizer_and_muon_overrides_are_forwarded_exactly(self):
        args = parse_args(
            "--optimizer",
            "muon",
            "--muon-adjust-lr-fn",
            "original",
            "--muon-ns-steps",
            "7",
            "--muon-momentum",
            "0.0",
            "--no-muon-nesterov",
            "--no-muon-qkv-split",
        )

        for command in (
            run_pipeline.train_command(args),
            run_pipeline.sft_command(args),
        ):
            self.assertIn("--optimizer", command)
            self.assertEqual(command[command.index("--optimizer") + 1], "muon")
            self.assertEqual(
                command[command.index("--muon-adjust-lr-fn") + 1], "original"
            )
            self.assertEqual(command[command.index("--muon-ns-steps") + 1], "7")
            self.assertEqual(command[command.index("--muon-momentum") + 1], "0.0")
            self.assertIn("--no-muon-nesterov", command)
            self.assertIn("--no-muon-qkv-split", command)

    def test_explicit_adamw_override_does_not_invent_muon_flags(self):
        args = parse_args("--optimizer", "adamw")

        for command in (
            run_pipeline.train_command(args),
            run_pipeline.sft_command(args),
        ):
            self.assertEqual(command[command.index("--optimizer") + 1], "adamw")
            self.assertTrue(
                (_OPTIMIZER_FLAGS - {"--optimizer"}).isdisjoint(command), command
            )


if __name__ == "__main__":
    unittest.main()
