import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.run_pretrain_ablations import build_commands, main, parse_args


class PretrainAblationTests(unittest.TestCase):
    def test_default_matrix_is_three_lrs_by_two_schedules(self):
        args = parse_args([])
        commands = build_commands(args)

        self.assertEqual(len(commands), 6)
        self.assertEqual(len({tuple(command) for command in commands}), 6)
        self.assertTrue(all("--target_tokens" in command for command in commands))
        self.assertTrue(all("--pretrain-lr" in command for command in commands))

    def test_torch_matrix_includes_device(self):
        args = parse_args([
            "--backend", "torch",
            "--device", "cpu",
            "--learning-rates", "0.0004",
            "--schedules", "cosine",
        ])
        command = build_commands(args)[0]

        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_execute_refuses_to_overwrite_existing_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            occupied = Path(temp_dir) / "300m" / "cosine-lr-4e-04"
            occupied.mkdir(parents=True)
            (occupied / "checkpoint.safetensors").write_text("existing")
            with mock.patch("scripts.run_pretrain_ablations.subprocess.run") as run:
                result = main([
                    "--learning-rates", "0.0004",
                    "--schedules", "cosine",
                    "--output-root", temp_dir,
                    "--execute",
                ])

        self.assertEqual(result, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
