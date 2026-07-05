"""Small JSON status writer for local training monitors."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_FILENAME = "training_status.json"
STATUS_HISTORY_FILENAME = "training_history.jsonl"
DEFAULT_MONITOR_PORT = 8765


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    os.replace(tmp, target)


def status_path_for_checkpoint_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, STATUS_FILENAME)


def history_path_for_checkpoint_dir(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, STATUS_HISTORY_FILENAME)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def _is_disabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def _port_is_listening(port: int) -> bool:
    targets = (
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET6, ("::1", port, 0, 0)),
    )
    for family, address in targets:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                if sock.connect_ex(address) == 0:
                    return True
            except OSError:
                continue
    return False


def _monitor_port() -> int:
    value = os.environ.get("SPAKIE_MONITOR_PORT", "")
    if not value:
        return DEFAULT_MONITOR_PORT
    try:
        port = int(value)
    except ValueError:
        return DEFAULT_MONITOR_PORT
    return port if 0 < port < 65536 else DEFAULT_MONITOR_PORT


def start_background_monitor(status_file: str, checkpoint_dir: str) -> dict[str, Any] | None:
    """Start the LAN monitor in a detached subprocess unless it is disabled/running."""
    if _is_disabled(os.environ.get("SPAKIE_MONITOR")):
        return None

    port = _monitor_port()
    url = f"http://{lan_ip()}:{port}"
    password_configured = bool(os.environ.get("MONITOR_PASSWORD", "").strip())
    if _port_is_listening(port):
        return {
            "started": False,
            "url": url,
            "port": port,
            "reason": "already_listening",
            "password_configured": password_configured,
        }

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "monitor_training.py"
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    default_checkpoint_root = repo_root / "checkpoints"
    try:
        checkpoint_path.resolve().relative_to(default_checkpoint_root.resolve())
        search_root = default_checkpoint_root
    except ValueError:
        search_root = checkpoint_path
    log_path = Path(checkpoint_dir) / "monitor_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable or "python3",
        str(script_path),
        "--host",
        "::",
        "--port",
        str(port),
        "--checkpoint-dir",
        str(search_root),
    ]

    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                close_fds=True,
                start_new_session=True,
            )
    except OSError as exc:
        return {
            "started": False,
            "url": url,
            "port": port,
            "log_path": str(log_path),
            "reason": f"failed: {exc}",
        }

    return {
        "started": True,
        "url": url,
        "port": port,
        "pid": process.pid,
        "process": process,
        "log_path": str(log_path),
        "status_file": status_file,
        "search_root": str(search_root),
        "password_configured": password_configured,
    }


def format_monitor_start_message(info: dict[str, Any] | None) -> str:
    if info is None:
        return "Monitor autostart disabled."
    if info.get("started"):
        return f"Monitor: {info['url']} (background, log: {info['log_path']})"
    reason = info.get("reason", "not started")
    if reason == "already_listening":
        if info.get("password_configured"):
            return f"Monitor: {info['url']} (already running; restart it if it was started without a password)"
        return f"Monitor: {info['url']} (already running)"
    return f"Monitor not started ({reason}); log: {info.get('log_path', 'n/a')}"


def stop_background_monitor(info: dict[str, Any] | None, *, timeout: float = 3.0) -> bool:
    """Stop the monitor subprocess owned by this training process, if any."""
    if not info or not info.get("started"):
        return False
    process = info.get("process")
    if not isinstance(process, subprocess.Popen):
        return False
    if process.poll() is not None:
        return False

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
    return True


class TrainingStatusWriter:
    """Rate-limited status.json writer shared by pretrain and SFT loops."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        stage: str,
        backend: str,
        preset: str,
        total_steps: int = 0,
        target_tokens: int = 0,
        elapsed_offset: float = 0.0,
        write_interval: float = 5.0,
    ) -> None:
        self.path = status_path_for_checkpoint_dir(checkpoint_dir)
        self.history_path = history_path_for_checkpoint_dir(checkpoint_dir)
        self.write_interval = max(float(write_interval), 0.0)
        self.started_at = iso_now()
        self.start_time = time.time() - max(float(elapsed_offset), 0.0)
        self.last_write = 0.0
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "stage": stage,
            "backend": backend,
            "preset": preset,
            "status": "starting",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "checkpoint_dir": checkpoint_dir,
            "started_at": self.started_at,
            "updated_at": self.started_at,
            "elapsed_seconds": 0.0,
            "step": 0,
            "total_steps": int(total_steps or 0),
            "target_tokens": int(target_tokens or 0),
            "message": "",
        }
        self.update(force=True)

    def update(self, *, force: bool = False, **fields: Any) -> None:
        now = time.time()
        self.payload.update({key: value for key, value in fields.items() if value is not None})
        self.payload["updated_at"] = iso_now()
        self.payload["elapsed_seconds"] = max(0.0, now - self.start_time)
        self._refresh_derived_fields()
        if force or self.last_write == 0.0 or now - self.last_write >= self.write_interval:
            atomic_write_json(self.path, self.payload)
            self._append_history_point()
            self.last_write = now

    def due(self) -> bool:
        return time.time() - self.last_write >= self.write_interval

    def finish(self, status: str, *, message: str = "", **fields: Any) -> None:
        self.update(force=True, status=status, message=message, **fields)

    def _append_history_point(self) -> None:
        if "train_loss" not in self.payload and "val_loss" not in self.payload:
            return
        point = {
            "updated_at": self.payload.get("updated_at"),
            "step": self.payload.get("step"),
            "train_loss": self.payload.get("train_loss"),
            "val_loss": self.payload.get("val_loss"),
        }
        target = Path(self.history_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            json.dump(point, handle, sort_keys=True, default=_json_default)
            handle.write("\n")

    def _refresh_derived_fields(self) -> None:
        step = int(self.payload.get("step") or 0)
        total_steps = int(self.payload.get("total_steps") or 0)
        tokens = int(self.payload.get("tokens_processed") or 0)
        target_tokens = int(self.payload.get("target_tokens") or 0)
        elapsed = float(self.payload.get("elapsed_seconds") or 0.0)

        if total_steps > 0:
            self.payload["step_progress"] = min(max(step / total_steps, 0.0), 1.0)
        if target_tokens > 0:
            self.payload["token_progress"] = min(max(tokens / target_tokens, 0.0), 1.0)
        if elapsed > 0 and tokens > 0:
            self.payload["tok_per_sec"] = tokens / elapsed

        progress = float(self.payload.get("token_progress") or self.payload.get("step_progress") or 0.0)
        if elapsed > 0 and progress > 0:
            self.payload["eta_seconds"] = max(0.0, elapsed * (1.0 - progress) / progress)
