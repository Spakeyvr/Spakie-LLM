"""Shared Muon optimizer constants and small pure-Python helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


OPTIMIZER_CHOICES = ("muon", "adamw")
MUON_ADJUST_LR_CHOICES = ("match_rms_adamw", "original", "none")
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
# DeepSeek-V4 hybrid Newton-Schulz: the final iterations use the classic cubic
# f(x) = 2x - 1.5x^3 + 0.5x^5 (f(1) = 1, f'(1) = 0), which converges
# quadratically and pins singular values at 1. The quintic above expands small
# singular values quickly but only settles into an oscillating ~[0.7, 1.2]
# band, never at 1.
MUON_NS_POLISH_COEFFICIENTS = (2.0, -1.5, 0.5)
# Parity limits between the torch and MLX Newton-Schulz outputs. On M5-class
# GPUs MLX routes fp32 GEMMs through the neural-accelerator ("nax") kernels,
# which accumulate at reduced precision (~1e-3 relative per matmul vs ~1e-7
# for true fp32); over muon_ns_steps iterations that legitimately drifts the
# elementwise output by up to ~7e-3 while orthogonalization quality stays
# identical to torch. The limits below must tolerate that hardware drift; a
# real implementation bug (wrong coefficients, transpose, normalization)
# produces O(1e-1) divergence and is still caught.
MUON_FP32_MAX_ABS = 1e-2
MUON_FP32_MAX_REL = 5e-2
MUON_BF16_MAX_ABS = 2e-2
MUON_BF16_MAX_REL = 5e-2
MUON_OPTIMIZER_SCHEMA_VERSION = 2


class MuonPrecomputeError(RuntimeError):
    """A Muon transform failed before any parameter or optimizer mutation."""

    safe_to_fallback = True


@dataclass(frozen=True)
class MuonSettings:
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS
    eps: float = 1e-7
    adjust_lr_fn: str = "match_rms_adamw"
    qkv_split: bool = True
    ns_polish_steps: int = 0
    split_gate_up: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "ns_steps": self.ns_steps,
            "ns_coefficients": list(self.ns_coefficients),
            "eps": self.eps,
            "adjust_lr_fn": self.adjust_lr_fn,
            "qkv_split": self.qkv_split,
            "ns_polish_steps": self.ns_polish_steps,
            "split_gate_up": self.split_gate_up,
        }


def normalize_optimizer_kind(kind: str | None) -> str:
    value = (kind or "muon").lower()
    if value not in OPTIMIZER_CHOICES:
        raise ValueError(f"Unsupported optimizer '{kind}'. Choices: {', '.join(OPTIMIZER_CHOICES)}")
    return value


def normalize_muon_adjust_lr_fn(name: str | None) -> str:
    value = (name or "match_rms_adamw").lower()
    if value not in MUON_ADJUST_LR_CHOICES:
        raise ValueError(
            f"Unsupported Muon LR adjustment '{name}'. Choices: {', '.join(MUON_ADJUST_LR_CHOICES)}"
        )
    return value


def muon_settings_from_config(config) -> MuonSettings:
    coeffs = getattr(config, "muon_ns_coefficients", MUON_NS_COEFFICIENTS)
    # Validate the schedule here so a bad CLI override fails at startup rather
    # than inside the Muon update, where it would look like a fallback-safe error.
    muon_ns_step_coefficients(
        int(getattr(config, "muon_ns_steps", 5)),
        tuple(float(x) for x in coeffs),
        int(getattr(config, "muon_ns_polish_steps", 0)),
    )
    return MuonSettings(
        momentum=float(getattr(config, "muon_momentum", 0.95)),
        nesterov=bool(getattr(config, "muon_nesterov", True)),
        ns_steps=int(getattr(config, "muon_ns_steps", 5)),
        ns_coefficients=tuple(float(x) for x in coeffs),
        eps=float(getattr(config, "muon_eps", 1e-7)),
        adjust_lr_fn=normalize_muon_adjust_lr_fn(getattr(config, "muon_adjust_lr_fn", "match_rms_adamw")),
        qkv_split=bool(getattr(config, "muon_qkv_split", True)),
        ns_polish_steps=int(getattr(config, "muon_ns_polish_steps", 0)),
        split_gate_up=bool(getattr(config, "muon_split_gate_up", False)),
    )


def muon_ns_step_coefficients(
    ns_steps: int,
    ns_coefficients: tuple[float, float, float] = MUON_NS_COEFFICIENTS,
    ns_polish_steps: int = 0,
) -> tuple[tuple[float, float, float], ...]:
    """Per-iteration (a, b, c): the last ``ns_polish_steps`` use the polish cubic."""
    if not 0 <= ns_polish_steps <= ns_steps:
        raise ValueError("ns_polish_steps must be between 0 and ns_steps")
    fast = tuple(float(x) for x in ns_coefficients)
    return (fast,) * (ns_steps - ns_polish_steps) + (MUON_NS_POLISH_COEFFICIENTS,) * ns_polish_steps


def adjusted_muon_lr(lr: float, shape: tuple[int, int], adjust_lr_fn: str) -> float:
    rows, cols = int(shape[0]), int(shape[1])
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * lr * math.sqrt(max(rows, cols))
    if adjust_lr_fn == "original":
        return lr * math.sqrt(max(1.0, rows / max(cols, 1)))
    if adjust_lr_fn == "none":
        return lr
    raise ValueError(f"Unsupported Muon LR adjustment '{adjust_lr_fn}'")


def is_muon_parameter_name(name: str, ndim: int) -> bool:
    """Return whether a parameter should receive Muon instead of auxiliary AdamW."""
    if ndim != 2:
        return False
    if name in {"tok_emb.weight", "lm_head.weight"}:
        return False
    if name.endswith(".bias") or ".ln" in name or "norm" in name.lower():
        return False
    if name.endswith(".weight") and any(part in name for part in (".attn.", ".mlp.")):
        return True
    return False


def muon_projection_split_count(name: str) -> int:
    """Number of independent projections fused in ``name``."""
    if name.endswith("attn.qkv.weight"):
        return 3
    if name.endswith("attn.kv_proj.weight"):
        return 2
    if name.endswith("mlp.gate_up.weight"):
        return 2
    return 1


def muon_update_split_count(name: str, settings: MuonSettings) -> int:
    """Row blocks of ``name`` that Muon orthogonalizes as separate matrices.

    Fused attention projections follow ``qkv_split``; the fused SwiGLU
    gate/up projection follows ``split_gate_up``.
    """
    enabled = settings.split_gate_up if name.endswith("mlp.gate_up.weight") else settings.qkv_split
    return muon_projection_split_count(name) if enabled else 1


def adamw_fallback_warning(stage: str) -> str:
    return (
        f"WARNING: {stage} is using AdamW as an explicit fallback. "
        "Muon is the required default and strongly preferred optimizer for Spakie training."
    )


def optimizer_kind(optimizer, config, *, stage: str) -> str:
    """Return the active optimizer kind for pretrain or SFT."""
    field = "pretrain_optimizer" if stage == "pretrain" else "sft_optimizer"
    return str(getattr(optimizer, "optimizer_kind", getattr(config, field, "muon")))


def should_adamw_fallback(exc: BaseException, optimizer, config, *, stage: str, allow: bool) -> bool:
    return (
        allow
        and optimizer_kind(optimizer, config, stage=stage) == "muon"
        and bool(getattr(exc, "safe_to_fallback", False))
        and not isinstance(exc, KeyboardInterrupt)
    )
