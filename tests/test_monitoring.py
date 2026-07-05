import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts.monitor_training import (
    _auth_token,
    _password_value,
    _uses_ipv6_host,
    create_server,
    find_latest_status,
    load_history,
    load_status,
    run_prompt_generation,
    select_prompt_checkpoint,
)
from training.monitor import (
    STATUS_FILENAME,
    TrainingStatusWriter,
    atomic_write_json,
    start_background_monitor,
    stop_background_monitor,
)


class TrainingMonitorTests(unittest.TestCase):
    def test_atomic_write_json_creates_parent_and_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "status.json"
            atomic_write_json(path, {"step": 3, "loss": 1.25})

            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["step"], 3)
        self.assertEqual(payload["loss"], 1.25)

    def test_status_writer_derives_progress_and_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TrainingStatusWriter(
                tmp,
                stage="pretrain",
                backend="torch",
                preset="92m",
                total_steps=10,
                target_tokens=100,
                write_interval=999,
            )
            writer.start_time = time.time() - 10
            writer.update(force=True, status="running", step=5, tokens_processed=50)

            status_path = Path(tmp) / STATUS_FILENAME
            with status_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["status"], "running")
        self.assertAlmostEqual(payload["step_progress"], 0.5)
        self.assertAlmostEqual(payload["token_progress"], 0.5)
        self.assertGreater(payload["tok_per_sec"], 0)
        self.assertGreater(payload["eta_seconds"], 0)

    def test_status_writer_appends_raw_loss_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TrainingStatusWriter(
                tmp,
                stage="pretrain",
                backend="mlx",
                preset="180m",
                total_steps=10,
                write_interval=999,
            )
            writer.update(force=True, status="running", step=4, train_loss=2.5, val_loss=2.8)
            history = load_history(Path(tmp) / STATUS_FILENAME, Path(tmp))

        self.assertTrue(history["ok"])
        self.assertEqual(len(history["history"]), 1)
        self.assertEqual(history["history"][0]["step"], 4)
        self.assertEqual(history["history"][0]["train_loss"], 2.5)
        self.assertEqual(history["history"][0]["val_loss"], 2.8)

    def test_find_latest_status_and_load_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "92m" / STATUS_FILENAME
            newer = root / "300m" / STATUS_FILENAME
            atomic_write_json(older, {"step": 1})
            time.sleep(0.01)
            atomic_write_json(newer, {"step": 2})

            self.assertEqual(find_latest_status(root), newer)
            payload = load_status(None, root)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"]["step"], 2)

    def test_select_prompt_checkpoint_prefers_status_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "custom_best.safetensors"
            stage_best = root / "pretrain_best.safetensors"
            best.touch()
            stage_best.touch()

            checkpoint, source = select_prompt_checkpoint(
                {
                    "backend": "mlx",
                    "stage": "pretrain",
                    "checkpoint_dir": str(root),
                    "best_checkpoint": str(best),
                }
            )

        self.assertEqual(checkpoint, best)
        self.assertEqual(source, "best_checkpoint")

    def test_select_prompt_checkpoint_uses_stage_best_for_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "sft_best.pt"
            best.touch()

            checkpoint, source = select_prompt_checkpoint(
                {
                    "backend": "torch",
                    "stage": "sft",
                    "checkpoint_dir": str(root),
                }
            )

        self.assertEqual(checkpoint, best)
        self.assertEqual(source, "stage_best")

    def test_run_prompt_generation_invokes_chat_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "pretrain_best.safetensors"
            best.touch()
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "backend": "mlx",
                        "preset": "180m",
                        "mode": "continue",
                        "checkpoint": str(best),
                        "response": "hello",
                    }
                ),
                stderr="",
            )

            with mock.patch("scripts.monitor_training.subprocess.run", return_value=completed) as run:
                payload = run_prompt_generation(
                    {
                        "ok": True,
                        "status": {
                            "backend": "mlx",
                            "stage": "pretrain",
                            "preset": "180m",
                            "checkpoint_dir": str(root),
                        },
                    },
                    "Hello",
                )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["response"], "hello")
        self.assertEqual(payload["checkpoint_source"], "stage_best")
        command = run.call_args.args[0]
        self.assertIn("chat_once.py", command[1])
        self.assertIn("--backend", command)
        self.assertIn("--checkpoint", command)
        self.assertIn(str(best), command)

    def test_background_monitor_respects_disable_env(self):
        old_value = os.environ.get("SPAKIE_MONITOR")
        os.environ["SPAKIE_MONITOR"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = start_background_monitor(str(Path(tmp) / STATUS_FILENAME), tmp)
        finally:
            if old_value is None:
                os.environ.pop("SPAKIE_MONITOR", None)
            else:
                os.environ["SPAKIE_MONITOR"] = old_value

        self.assertIsNone(result)

    def test_background_monitor_detects_existing_listener(self):
        old_port = os.environ.get("SPAKIE_MONITOR_PORT")
        os.environ["SPAKIE_MONITOR_PORT"] = "6543"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with mock.patch("training.monitor.lan_ip", return_value="127.0.0.1"):
                    with mock.patch("training.monitor._port_is_listening", return_value=True):
                        result = start_background_monitor(str(Path(tmp) / STATUS_FILENAME), tmp)
        finally:
            if old_port is None:
                os.environ.pop("SPAKIE_MONITOR_PORT", None)
            else:
                os.environ["SPAKIE_MONITOR_PORT"] = old_port

        self.assertIsNotNone(result)
        self.assertFalse(result["started"])
        self.assertEqual(result["reason"], "already_listening")

    def test_stop_background_monitor_terminates_owned_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            stopped = stop_background_monitor({"started": True, "process": process}, timeout=1.0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1.0)

        self.assertTrue(stopped)
        self.assertIsNotNone(process.poll())

    def test_auth_token_is_secret_scoped(self):
        first = _auth_token("password", "secret-one")
        second = _auth_token("password", "secret-two")

        self.assertEqual(first, _auth_token("password", "secret-one"))
        self.assertNotEqual(first, second)

    def test_password_value_strips_empty_values(self):
        self.assertEqual(_password_value("  secret  "), "secret")
        self.assertEqual(_password_value("   "), "")
        self.assertEqual(_password_value(None), "")

    def test_ipv6_host_detection(self):
        self.assertTrue(_uses_ipv6_host("::"))
        self.assertTrue(_uses_ipv6_host("2001:db8::1"))
        self.assertFalse(_uses_ipv6_host("0.0.0.0"))
        self.assertFalse(_uses_ipv6_host("127.0.0.1"))

    def test_create_server_uses_ipv6_class_for_ipv6_host(self):
        class Handler:
            pass

        with mock.patch("scripts.monitor_training.ThreadingHTTPServer") as ipv4_server:
            with mock.patch("scripts.monitor_training.DualStackThreadingHTTPServer") as ipv6_server:
                create_server("::", 8765, Handler)
                ipv6_server.assert_called_once_with(("::", 8765), Handler)
                ipv4_server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
