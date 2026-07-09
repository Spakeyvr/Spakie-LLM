"""Shared Muon optimizer constants and small pure-Python helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


OPTIMIZER_CHOICES = ("muon", "adamw")
MUON_ADJUST_LR_CHOICES = ("match_rms_adamw", "original", "none")
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
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

    def as_dict(self) -> dict[str, object]:
        return {
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "ns_steps": self.ns_steps,
            "ns_coefficients": list(self.ns_coefficients),
            "eps": self.eps,
            "adjust_lr_fn": self.adjust_lr_fn,
            "qkv_split": self.qkv_split,
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
    return MuonSettings(
        momentum=float(getattr(config, "muon_momentum", 0.95)),
        nesterov=bool(getattr(config, "muon_nesterov", True)),
        ns_steps=int(getattr(config, "muon_ns_steps", 5)),
        ns_coefficients=tuple(float(x) for x in coeffs),
        eps=float(getattr(config, "muon_eps", 1e-7)),
        adjust_lr_fn=normalize_muon_adjust_lr_fn(getattr(config, "muon_adjust_lr_fn", "match_rms_adamw")),
        qkv_split=bool(getattr(config, "muon_qkv_split", True)),
    )


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
    if name in {"tok_emb.weight", "pos_emb.weight", "lm_head.weight"}:
        return False
    if name.endswith(".bias") or ".ln" in name or "norm" in name.lower():
        return False
    if name.endswith(".weight") and any(part in name for part in (".attn.", ".mlp.")):
        return True
    return False


def is_qkv_parameter_name(name: str) -> bool:
    return name.endswith("attn.qkv.weight")


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
