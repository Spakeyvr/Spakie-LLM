"""Serve the latest Spakie training status over a small LAN web page."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.cookies import SimpleCookie
import hmac
import json
import os
from secrets import token_hex
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.monitor import STATUS_FILENAME, STATUS_HISTORY_FILENAME, lan_ip


AUTH_COOKIE_NAME = "spakie_monitor_auth"


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spakie Training</title>
<style>
:root { color-scheme: dark; --bg: #101316; --panel: #1a2026; --panel-soft: #151b21; --muted: #91a1ae; --text: #f4f7f9; --ok: #47d18c; --warn: #ffcf5a; --bad: #ff6b6b; --line: #2b3540; --track: #0c0f12; --train: #49d69a; --val: #67b7ff; }
[data-theme="light"] { color-scheme: light; --bg: #f5f7fa; --panel: #ffffff; --panel-soft: #f0f4f8; --muted: #627386; --text: #101820; --ok: #148a58; --warn: #9a6a00; --bad: #be3b3b; --line: #d4dde7; --track: #e7edf3; --train: #138a5a; --val: #166fc5; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
main { width: min(920px, 100%); margin: 0 auto; padding: 18px 14px 28px; }
header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 16px; }
h1 { margin: 0; font-size: 1.55rem; font-weight: 720; }
.sub { color: var(--muted); font-size: .92rem; margin-top: 4px; overflow-wrap: anywhere; }
.headerTools { display: flex; align-items: center; gap: 8px; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: .82rem; color: var(--muted); white-space: nowrap; }
.themeToggle { width: 34px; height: 34px; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); color: var(--text); cursor: pointer; font-size: 1rem; line-height: 1; display: grid; place-items: center; }
.running { color: var(--ok); }
.stopped, .interrupted { color: var(--warn); }
.error { color: var(--bad); }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 10px; }
.metricGrid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 84px; }
.wide { grid-column: 1 / -1; }
.label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.value { font-size: 1.55rem; font-weight: 720; margin-top: 4px; overflow-wrap: anywhere; }
.small { font-size: .92rem; color: var(--muted); margin-top: 5px; overflow-wrap: anywhere; }
.bar { height: 12px; border-radius: 999px; background: var(--track); overflow: hidden; margin-top: 10px; border: 1px solid var(--line); }
.fill { height: 100%; width: 0%; background: linear-gradient(90deg, #47d18c, #6cb7ff); transition: width .25s ease; }
.chartHead { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 10px; }
.legend { display: flex; gap: 12px; color: var(--muted); font-size: .84rem; white-space: nowrap; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.swatch { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
.swatch.train { background: var(--train); }
.swatch.val { background: var(--val); }
.chartWrap { position: relative; height: 220px; border-radius: 8px; background: var(--panel-soft); border: 1px solid var(--line); overflow: hidden; }
canvas { width: 100%; height: 100%; display: block; }
textarea { width: 100%; min-height: 92px; resize: vertical; border: 1px solid var(--line); border-radius: 7px; background: var(--panel-soft); color: var(--text); padding: 10px 12px; font: inherit; line-height: 1.35; }
button.sendPrompt { min-height: 38px; border: 1px solid var(--line); border-radius: 7px; background: var(--text); color: var(--bg); font: inherit; font-weight: 720; padding: 8px 13px; cursor: pointer; }
button.sendPrompt:disabled { cursor: wait; opacity: .6; }
.promptActions { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 10px; }
.promptResponse { margin-top: 12px; padding: 12px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel-soft); min-height: 54px; }
pre { margin: 0; white-space: pre-wrap; color: var(--muted); font-size: .84rem; line-height: 1.45; }
@media (max-width: 820px) { .metricGrid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .grid, .metricGrid { grid-template-columns: 1fr; } header { display: block; } .headerTools { margin-top: 10px; } .chartWrap { height: 190px; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Spakie Training</h1>
      <div class="sub" id="path">Waiting for status...</div>
    </div>
    <div class="headerTools">
      <button class="themeToggle" id="themeToggle" type="button" title="Toggle light/dark mode" aria-label="Toggle light/dark mode">D</button>
      <div class="pill" id="status">connecting</div>
    </div>
  </header>
  <section class="grid">
    <div class="card wide">
      <div class="label">Progress</div>
      <div class="value" id="progress">--</div>
      <div class="bar"><div class="fill" id="progressFill"></div></div>
      <div class="small" id="progressDetail"></div>
    </div>
  </section>
  <section class="grid metricGrid">
    <div class="card">
      <div class="label">Train Loss</div>
      <div class="value" id="trainLoss">--</div>
    </div>
    <div class="card">
      <div class="label">Val / Best</div>
      <div class="value" id="valLoss">--</div>
      <div class="small" id="bestLoss"></div>
    </div>
    <div class="card">
      <div class="label">Throughput</div>
      <div class="value" id="throughput">--</div>
      <div class="small" id="lr"></div>
    </div>
    <div class="card">
      <div class="label">ETA</div>
      <div class="value" id="eta">--</div>
      <div class="small" id="elapsed"></div>
    </div>
  </section>
  <section class="grid">
    <div class="card wide">
      <div class="chartHead">
        <div>
          <div class="label">Raw Loss</div>
          <div class="small" id="chartMeta">waiting for points</div>
        </div>
        <div class="legend">
          <span><i class="swatch train"></i>train_loss</span>
          <span><i class="swatch val"></i>val_loss</span>
        </div>
      </div>
      <div class="chartWrap">
        <canvas id="lossChart"></canvas>
      </div>
    </div>
    <div class="card wide">
      <div class="chartHead">
        <div>
          <div class="label">Prompt Checkpoint</div>
          <div class="small" id="promptMeta">best checkpoint, auto mode</div>
        </div>
      </div>
      <textarea id="promptInput" maxlength="8000" placeholder="Type a prompt..."></textarea>
      <div class="promptActions">
        <button class="sendPrompt" id="promptSend" type="button">Send</button>
        <div class="small" id="promptState">idle</div>
      </div>
      <div class="promptResponse">
        <pre id="promptOutput">No response yet.</pre>
      </div>
    </div>
    <div class="card wide">
      <div class="label">Details</div>
      <pre id="details">No status yet.</pre>
    </div>
  </section>
</main>
<script>
const historyPoints = [];
let historyLoaded = false;
const root = document.documentElement;
const savedTheme = localStorage.getItem("spakieMonitorTheme") || "dark";
root.dataset.theme = savedTheme;

function fmtNumber(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "--";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
}
function fmtLoss(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "--";
  return Number(n).toFixed(4);
}
function fmtDuration(seconds) {
  if (!seconds || seconds < 0) return "--";
  seconds = Math.round(seconds);
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
function setText(id, value) { document.getElementById(id).textContent = value; }
function cssVar(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
function addPoint(s) {
  if (!s || s.step === undefined) return;
  const point = {
    key: `${s.step}|${s.updated_at || ""}`,
    step: Number(s.step),
    train_loss: Number.isFinite(Number(s.train_loss)) ? Number(s.train_loss) : null,
    val_loss: Number.isFinite(Number(s.val_loss)) ? Number(s.val_loss) : null,
    updated_at: s.updated_at || "",
  };
  if (point.train_loss === null && point.val_loss === null) return;
  if (historyPoints.some((item) => item.key === point.key)) return;
  historyPoints.push(point);
  historyPoints.sort((a, b) => a.step - b.step);
  if (historyPoints.length > 2000) historyPoints.splice(0, historyPoints.length - 2000);
}
function loadHistoryPoints(points) {
  for (const row of points || []) {
    addPoint(row);
  }
}
async function refreshHistory() {
  if (historyLoaded) return;
  try {
    const res = await fetch("/api/history", { cache: "no-store" });
    if (res.status === 401) return;
    const data = await res.json();
    if (data.ok) loadHistoryPoints(data.history);
    historyLoaded = true;
  } catch (err) {
    historyLoaded = true;
  }
}
function drawChart() {
  const canvas = document.getElementById("lossChart");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { left: 48, right: 14, top: 16, bottom: 34 };
  const plotW = Math.max(1, w - pad.left - pad.right);
  const plotH = Math.max(1, h - pad.top - pad.bottom);
  const line = cssVar("--line");
  const muted = cssVar("--muted");
  const text = cssVar("--text");
  const trainColor = cssVar("--train");
  const valColor = cssVar("--val");
  const values = [];
  for (const point of historyPoints) {
    if (point.train_loss !== null) values.push(point.train_loss);
    if (point.val_loss !== null) values.push(point.val_loss);
  }
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.lineWidth = 1;
  ctx.strokeStyle = line;
  ctx.fillStyle = muted;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }
  if (historyPoints.length === 0 || values.length === 0) {
    ctx.fillStyle = muted;
    ctx.fillText("Waiting for raw loss points...", pad.left, pad.top + 24);
    setText("chartMeta", "waiting for points");
    return;
  }
  let minY = Math.min(...values);
  let maxY = Math.max(...values);
  if (minY === maxY) {
    minY -= 0.01;
    maxY += 0.01;
  }
  const yPad = (maxY - minY) * 0.08;
  minY -= yPad;
  maxY += yPad;
  const minStep = historyPoints[0].step;
  const maxStep = historyPoints[historyPoints.length - 1].step;
  const stepSpan = Math.max(1, maxStep - minStep);
  const xFor = (step) => pad.left + ((step - minStep) / stepSpan) * plotW;
  const yFor = (value) => pad.top + (1 - (value - minY) / (maxY - minY)) * plotH;

  ctx.fillStyle = muted;
  ctx.fillText(maxY.toFixed(3), 6, pad.top + 4);
  ctx.fillText(minY.toFixed(3), 6, pad.top + plotH);
  ctx.fillText(`step ${fmtNumber(minStep)} -> ${fmtNumber(maxStep)}`, pad.left, h - 10);

  function drawSeries(field, color) {
    const pts = historyPoints.filter((point) => point[field] !== null);
    if (pts.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((point, idx) => {
      const x = xFor(point.step);
      const y = yFor(point[field]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const latest = pts[pts.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(xFor(latest.step), yFor(latest[field]), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  drawSeries("train_loss", trainColor);
  drawSeries("val_loss", valColor);
  ctx.fillStyle = text;
  setText("chartMeta", `${fmtNumber(historyPoints.length)} raw points, no smoothing`);
}
function applyTheme(theme) {
  root.dataset.theme = theme;
  localStorage.setItem("spakieMonitorTheme", theme);
  document.getElementById("themeToggle").textContent = theme === "dark" ? "D" : "L";
  drawChart();
}
async function refresh() {
  try {
    await refreshHistory();
    const res = await fetch("/api/status", { cache: "no-store" });
    const data = await res.json();
    if (!data.ok) {
      setText("status", "waiting");
      setText("path", data.message || "No training status found yet.");
      return;
    }
    const s = data.status;
    addPoint(s);
    drawChart();
    const statusEl = document.getElementById("status");
    statusEl.textContent = `${s.status || "unknown"} | ${s.stage || "?"}/${s.backend || "?"}`;
    statusEl.className = `pill ${s.status || ""}`;
    setText("path", `${s.preset || "unknown"} | ${s.checkpoint_dir || data.status_file || ""}`);

    const progress = s.token_progress ?? s.step_progress ?? 0;
    const pct = Math.max(0, Math.min(100, progress * 100));
    setText("progress", `${pct.toFixed(1)}%`);
    document.getElementById("progressFill").style.width = `${pct}%`;
    const stepPart = s.total_steps ? `step ${fmtNumber(s.step)} / ${fmtNumber(s.total_steps)}` : `step ${fmtNumber(s.step)}`;
    const tokenPart = s.target_tokens ? `tokens ${fmtNumber(s.tokens_processed)} / ${fmtNumber(s.target_tokens)}` : "";
    setText("progressDetail", [stepPart, tokenPart, s.message || ""].filter(Boolean).join(" | "));

    setText("trainLoss", fmtLoss(s.train_loss));
    setText("valLoss", fmtLoss(s.val_loss));
    setText("bestLoss", s.best_val_loss === null || s.best_val_loss === undefined ? "" : `best ${fmtLoss(s.best_val_loss)}`);
    setText("throughput", s.tok_per_sec ? `${fmtNumber(s.tok_per_sec)} tok/s` : "--");
    setText("lr", s.lr ? `lr ${Number(s.lr).toExponential(2)}` : "");
    setText("eta", fmtDuration(s.eta_seconds));
    setText("elapsed", `elapsed ${fmtDuration(s.elapsed_seconds)} | updated ${s.updated_at || "--"}`);

    const details = {
      epoch: s.epoch && s.epochs ? `${s.epoch}/${s.epochs}` : undefined,
      mlx_memory_gb: s.mlx_active_gb === undefined ? undefined : {
        active: Number(s.mlx_active_gb).toFixed(1),
        cache: Number(s.mlx_cache_gb).toFixed(1),
        peak: Number(s.mlx_peak_gb).toFixed(1),
      },
      best_checkpoint: s.best_checkpoint,
      last_checkpoint: s.last_checkpoint,
      status_file: data.status_file,
      pid: s.pid,
      host: s.host,
    };
    setText("details", JSON.stringify(details, null, 2));
  } catch (err) {
    const statusEl = document.getElementById("status");
    statusEl.textContent = "offline";
    statusEl.className = "pill error";
  }
}
async function sendPrompt() {
  const input = document.getElementById("promptInput");
  const button = document.getElementById("promptSend");
  const prompt = input.value.trim();
  if (!prompt) {
    setText("promptState", "enter a prompt first");
    return;
  }
  button.disabled = true;
  setText("promptState", "generating...");
  setText("promptOutput", "");
  try {
    const res = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    if (!data.ok) {
      setText("promptState", "error");
      setText("promptOutput", data.message || "Prompt failed.");
      return;
    }
    setText("promptState", `${data.mode || "auto"} | ${data.backend || "?"} | ${data.checkpoint_source || "checkpoint"}`);
    setText("promptMeta", data.checkpoint || "checkpoint used");
    setText("promptOutput", data.response || "");
  } catch (err) {
    setText("promptState", "offline");
    setText("promptOutput", "Could not reach the monitor prompt endpoint.");
  } finally {
    button.disabled = false;
  }
}
document.getElementById("themeToggle").addEventListener("click", () => {
  applyTheme(root.dataset.theme === "dark" ? "light" : "dark");
});
document.getElementById("promptSend").addEventListener("click", sendPrompt);
document.getElementById("promptInput").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    sendPrompt();
  }
});
window.addEventListener("resize", drawChart);
applyTheme(savedTheme);
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""

LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spakie Monitor Login</title>
<style>
:root { color-scheme: dark; --bg: #101316; --panel: #1a2026; --muted: #8fa0ad; --text: #f4f7f9; --line: #2b3540; --bad: #ff6b6b; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
main { width: min(380px, calc(100% - 28px)); background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; }
h1 { margin: 0 0 6px; font-size: 1.35rem; }
p { margin: 0 0 16px; color: var(--muted); }
input, button { width: 100%; min-height: 44px; border-radius: 7px; font: inherit; }
input { border: 1px solid var(--line); background: #0c0f12; color: var(--text); padding: 10px 12px; }
button { margin-top: 12px; border: 0; background: #47d18c; color: #07100b; font-weight: 700; }
.error { color: var(--bad); margin-top: 12px; min-height: 20px; }
</style>
</head>
<body>
<main>
  <h1>Spakie Monitor</h1>
  <p>Password required.</p>
  <form method="post" action="/login">
    <input name="password" type="password" autocomplete="current-password" autofocus>
    <button type="submit">Unlock</button>
  </form>
  <div class="error">{error}</div>
</main>
</body>
</html>
"""


def load_status(status_file: Path | None, search_root: Path) -> dict[str, Any]:
    selected = status_file if status_file is not None else find_latest_status(search_root)
    if selected is None:
        return {"ok": False, "message": f"No {STATUS_FILENAME} found under {search_root}"}
    try:
        with selected.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "status_file": str(selected), "message": f"Could not read status: {exc}"}
    return {"ok": True, "status_file": str(selected), "status": status}


def _selected_status_file(status_file: Path | None, search_root: Path) -> Path | None:
    return status_file if status_file is not None else find_latest_status(search_root)


def load_history(status_file: Path | None, search_root: Path, *, limit: int = 2000) -> dict[str, Any]:
    selected = _selected_status_file(status_file, search_root)
    if selected is None:
        return {"ok": False, "message": f"No {STATUS_FILENAME} found under {search_root}", "history": []}
    history_path = selected.with_name(STATUS_HISTORY_FILENAME)
    if not history_path.exists():
        return {"ok": True, "history_file": str(history_path), "history": []}

    rows: list[dict[str, Any]] = []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError as exc:
        return {"ok": False, "history_file": str(history_path), "message": f"Could not read history: {exc}", "history": []}

    if limit > 0 and len(rows) > limit:
        rows = rows[-limit:]
    return {"ok": True, "history_file": str(history_path), "history": rows}


def find_latest_status(root: Path) -> Path | None:
    candidates = [path for path in root.glob(f"**/{STATUS_FILENAME}") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _checkpoint_suffix(backend: str) -> str:
    return ".pt" if backend == "torch" else ".safetensors"


def _existing_checkpoint(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    return candidate if candidate.is_file() else None


def select_prompt_checkpoint(status: dict[str, Any]) -> tuple[Path | None, str]:
    """Pick the checkpoint the monitor prompt box should query."""
    backend = str(status.get("backend") or "mlx")
    stage = str(status.get("stage") or "pretrain")
    suffix = _checkpoint_suffix(backend)
    checkpoint_dir = Path(str(status.get("checkpoint_dir") or "checkpoints"))

    candidates: list[tuple[str, str | os.PathLike[str] | None]] = [
        ("best_checkpoint", status.get("best_checkpoint")),
        ("stage_best", checkpoint_dir / f"{stage}_best{suffix}"),
        ("pretrain_best", checkpoint_dir / f"pretrain_best{suffix}"),
        ("sft_best", checkpoint_dir / f"sft_best{suffix}"),
        ("last_checkpoint", status.get("last_checkpoint")),
        ("stage_interrupt", checkpoint_dir / f"{stage}_interrupt{suffix}"),
    ]
    for source, path in candidates:
        checkpoint = _existing_checkpoint(path)
        if checkpoint is not None:
            return checkpoint, source
    return None, ""


def _prompt_timeout() -> float:
    value = os.environ.get("SPAKIE_MONITOR_PROMPT_TIMEOUT", "").strip()
    if not value:
        return 180.0
    try:
        timeout = float(value)
    except ValueError:
        return 180.0
    return min(max(timeout, 5.0), 900.0)


def run_prompt_generation(
    status_payload: dict[str, Any],
    prompt: str,
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    if not status_payload.get("ok"):
        return {"ok": False, "message": status_payload.get("message", "No training status found.")}

    status = status_payload.get("status") or {}
    if not isinstance(status, dict):
        return {"ok": False, "message": "Training status is malformed."}

    prompt = prompt.strip()
    if not prompt:
        return {"ok": False, "message": "Prompt is empty."}
    if len(prompt) > 8000:
        return {"ok": False, "message": "Prompt is too long."}
    max_new_tokens = min(max(int(max_new_tokens or 256), 1), 1024)

    checkpoint, source = select_prompt_checkpoint(status)
    if checkpoint is None:
        return {"ok": False, "message": "No runnable checkpoint found yet."}

    backend = str(status.get("backend") or "mlx")
    preset = str(status.get("preset") or "")
    if not preset:
        return {"ok": False, "message": "Training status does not include a preset."}

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable or "python3",
        str(repo_root / "scripts" / "chat_once.py"),
        "--backend",
        backend,
        "--preset",
        preset,
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--precision",
        "auto",
    ]
    if backend == "torch":
        command.extend(["--device", "auto"])

    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=_prompt_timeout(),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Prompt generation timed out."}
    except OSError as exc:
        return {"ok": False, "message": f"Could not start prompt generation: {exc}"}

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        message = stdout
        try:
            parsed = json.loads(stdout)
            message = parsed.get("message", message)
        except json.JSONDecodeError:
            pass
        if completed.stderr.strip():
            message = f"{message}\n{completed.stderr.strip()}".strip()
        return {"ok": False, "message": message or "Prompt generation failed."}

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {"ok": False, "message": "Prompt generation returned malformed JSON."}
    if isinstance(payload, dict):
        payload["checkpoint_source"] = source
        return payload
    return {"ok": False, "message": "Prompt generation returned malformed payload."}


class DualStackThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self) -> None:
        if hasattr(socket, "IPV6_V6ONLY"):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        super().server_bind()


def _uses_ipv6_host(host: str) -> bool:
    return ":" in host


def _is_loopback_host(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"}


def create_server(host: str, port: int, handler) -> ThreadingHTTPServer:
    server_cls = DualStackThreadingHTTPServer if _uses_ipv6_host(host) else ThreadingHTTPServer
    return server_cls((host, port), handler)


def _auth_token(password: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), password.encode("utf-8"), "sha256").hexdigest()


def _password_value(value: str | None) -> str:
    return (value or "").strip()


def make_handler(status_file: Path | None, search_root: Path, *, verbose: bool, password: str = ""):
    password = _password_value(password)
    session_secret = token_hex(32)
    expected_token = _auth_token(password, session_secret) if password else ""
    prompt_lock = threading.Lock()

    class MonitorHandler(BaseHTTPRequestHandler):
        server_version = "SpakieTrainingMonitor/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if not self._is_authorized():
                if path.startswith("/api/"):
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"ok": False, "message": "password required"},
                    )
                else:
                    self._send_login()
                return
            if path == "/":
                self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if path == "/api/status":
                payload = load_status(status_file, search_root)
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                )
                return
            if path == "/api/history":
                payload = load_history(status_file, search_root)
                self._send(
                    200,
                    "application/json; charset=utf-8",
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                )
                return
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/prompt":
                if not self._is_authorized():
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"ok": False, "message": "password required"},
                    )
                    return
                self._handle_prompt()
                return
            if path != "/login":
                self._send(404, "text/plain; charset=utf-8", b"not found\n")
                return
            if not password:
                self._redirect("/")
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length > 4096:
                self._send_login(error=True)
                return
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            entered = parse_qs(body, keep_blank_values=True).get("password", [""])[0]
            if hmac.compare_digest(entered, password):
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", self._auth_cookie_header())
                self.send_header("Content-Length", "0")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self._send_login(error=True)

        def _handle_prompt(self) -> None:
            if not prompt_lock.acquire(blocking=False):
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "message": "another prompt is already running"},
                )
                return
            try:
                self._handle_prompt_exclusive()
            finally:
                prompt_lock.release()

        def _handle_prompt_exclusive(self) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0 or length > 65536:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "message": "Prompt request is empty or too large."},
                )
                return
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                request = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "message": "Prompt request must be JSON."},
                )
                return
            prompt = request.get("prompt") if isinstance(request, dict) else ""
            if not isinstance(prompt, str):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "message": "Prompt must be a string."},
                )
                return
            try:
                max_new_tokens = int(request.get("max_new_tokens", 256)) if isinstance(request, dict) else 256
            except (TypeError, ValueError):
                max_new_tokens = 256
            status_payload = load_status(status_file, search_root)
            payload = run_prompt_generation(status_payload, prompt, max_new_tokens=max_new_tokens)
            code = HTTPStatus.OK if payload.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(code, payload)

        def log_message(self, fmt: str, *args: Any) -> None:
            if verbose:
                super().log_message(fmt, *args)

        def _is_authorized(self) -> bool:
            if not password:
                return True
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get(AUTH_COOKIE_NAME)
            token = morsel.value if morsel is not None else ""
            return hmac.compare_digest(token, expected_token)

        def _auth_cookie_header(self) -> str:
            cookie = SimpleCookie()
            cookie[AUTH_COOKIE_NAME] = expected_token
            cookie[AUTH_COOKIE_NAME]["httponly"] = True
            cookie[AUTH_COOKIE_NAME]["path"] = "/"
            cookie[AUTH_COOKIE_NAME]["samesite"] = "Strict"
            return cookie.output(header="").strip()

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _send_login(self, *, error: bool = False) -> None:
            body = LOGIN_HTML.replace("{error}", "Wrong password." if error else "").encode("utf-8")
            self._send(HTTPStatus.UNAUTHORIZED, "text/html; charset=utf-8", body)

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            self._send(
                code,
                "application/json; charset=utf-8",
                json.dumps(payload, sort_keys=True).encode("utf-8"),
            )

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return MonitorHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve Spakie training status for phone monitoring")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; non-loopback hosts require --password")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory tree to scan for training_status.json")
    parser.add_argument("--status-file", default="", help="Exact training_status.json path to serve")
    parser.add_argument(
        "--password",
        default=os.environ.get("MONITOR_PASSWORD", ""),
        help="Require this password before serving the monitor (default: MONITOR_PASSWORD)",
    )
    parser.add_argument("--verbose", action="store_true", help="Log HTTP requests")
    args = parser.parse_args()

    if not _is_loopback_host(args.host) and not _password_value(args.password):
        parser.error("a non-loopback monitor bind requires --password or MONITOR_PASSWORD")

    status_file = Path(args.status_file).expanduser() if args.status_file else None
    search_root = Path(args.checkpoint_dir).expanduser()
    handler = make_handler(status_file, search_root, verbose=args.verbose, password=args.password)
    server = create_server(args.host, args.port, handler)
    server.daemon_threads = True

    host_for_print = lan_ip() if args.host in {"", "0.0.0.0", "::"} else args.host
    print(f"Serving Spakie training monitor at http://{host_for_print}:{args.port}")
    if _password_value(args.password):
        print("Password protection: enabled")
    else:
        print("Password protection: disabled (set MONITOR_PASSWORD or pass --password)")
    print("Open that URL on your phone while it is on the same Wi-Fi.")
    print("Press Ctrl+C to stop the monitor server.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping monitor server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
