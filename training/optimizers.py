"""PyTorch optimizer builders for AdamW fallback and Muon-first training."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime import RuntimeSettings, optimizer_kwargs
from training.muon_core import (
    MuonPrecomputeError,
    MuonSettings,
    adjusted_muon_lr,
    is_muon_parameter_name,
    is_qkv_parameter_name,
    muon_settings_from_config,
    normalize_optimizer_kind,
)


def muon_newton_schulz_torch(
    update: torch.Tensor,
    *,
    ns_steps: int = 5,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> torch.Tensor:
    if update.ndim != 2:
        raise ValueError("Muon Newton-Schulz expects a 2D tensor")
    a, b, c = ns_coefficients
    x = update.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(ns_steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=update.dtype)


class MuonAdamW:
    """Hybrid optimizer: Muon for hidden matrices, AdamW for auxiliary params."""

    optimizer_kind = "muon"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        weight_decay: float,
        runtime: RuntimeSettings,
        settings: MuonSettings,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.settings = settings
        self.muon_params: list[torch.nn.Parameter] = []
        self.muon_names: list[str] = []
        adamw_decay: list[torch.nn.Parameter] = []
        adamw_nodecay: list[torch.nn.Parameter] = []

        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            if is_muon_parameter_name(name, param.dim()):
                self.muon_names.append(name)
                self.muon_params.append(param)
            elif param.dim() >= 2:
                adamw_decay.append(param)
            else:
                adamw_nodecay.append(param)

        self.adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": weight_decay},
                {"params": adamw_nodecay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            **optimizer_kwargs(runtime),
        )
        self.state: dict[torch.nn.Parameter, dict[str, torch.Tensor]] = {}
        self.param_groups = [{"params": self.muon_params, "lr": lr, "weight_decay": weight_decay}] + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.muon_params:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
        self.adamw.zero_grad(set_to_none=set_to_none)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        self.param_groups[0]["lr"] = lr
        for group in self.adamw.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        # Build every Muon update before mutating either the model or optimizer
        # state.  In particular, Newton-Schulz can fail (or produce a shape
        # error) and the caller may explicitly fall back to AdamW.  Updating
        # auxiliary parameters first would make that fallback apply the same
        # gradients twice to those parameters.
        try:
            prepared: list[tuple[torch.nn.Parameter, torch.Tensor, list[tuple[int, torch.Tensor, float]]]] = []
            for name, param in zip(self.muon_names, self.muon_params, strict=True):
                grad = param.grad
                if grad is None:
                    continue
                state = self.state.get(param, {})
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(param, memory_format=torch.preserve_format)
                next_momentum = momentum_buffer.mul(self.settings.momentum).add(grad)
                update = grad.add(next_momentum, alpha=self.settings.momentum) if self.settings.nesterov else next_momentum
                chunks: list[tuple[int, torch.Tensor, float]] = []
                for start, chunk in self._update_chunks(name, update):
                    orthogonal = muon_newton_schulz_torch(
                        chunk,
                        ns_steps=self.settings.ns_steps,
                        ns_coefficients=self.settings.ns_coefficients,
                        eps=self.settings.eps,
                    )
                    chunk_lr = adjusted_muon_lr(self.lr, tuple(chunk.shape), self.settings.adjust_lr_fn)
                    chunks.append((start, orthogonal, chunk_lr))
                prepared.append((param, next_momentum, chunks))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise MuonPrecomputeError(f"Muon update preparation failed: {exc}") from exc

        self.adamw.step()
        for param, next_momentum, chunks in prepared:
            if self.weight_decay:
                param.data.mul_(1.0 - self.lr * self.weight_decay)
            for start, orthogonal, chunk_lr in chunks:
                param.data.narrow(0, start, orthogonal.shape[0]).add_(orthogonal, alpha=-chunk_lr)
            self.state.setdefault(param, {})["momentum_buffer"] = next_momentum

    def _update_chunks(self, name: str, update: torch.Tensor):
        if self.settings.qkv_split and is_qkv_parameter_name(name) and update.shape[0] % 3 == 0:
            chunk_size = update.shape[0] // 3
            for start in range(0, update.shape[0], chunk_size):
                yield start, update.narrow(0, start, chunk_size)
            return
        yield 0, update

    def state_dict(self) -> dict[str, object]:
        muon_state = {
            name: {"momentum_buffer": self.state[param]["momentum_buffer"]}
            for name, param in zip(self.muon_names, self.muon_params, strict=True)
            if param in self.state and "momentum_buffer" in self.state[param]
        }
        return {
            "optimizer_kind": self.optimizer_kind,
            "settings": self.settings.as_dict(),
            "muon_names": list(self.muon_names),
            "muon_state": muon_state,
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.adamw.load_state_dict(state_dict["adamw"])  # type: ignore[arg-type]
        saved_muon = state_dict.get("muon_state", {})
        if not isinstance(saved_muon, dict):
            return
        by_name = dict(zip(self.muon_names, self.muon_params, strict=True))
        for name, param_state in saved_muon.items():
            param = by_name.get(str(name))
            if param is None or not isinstance(param_state, dict):
                continue
            momentum = param_state.get("momentum_buffer")
            if isinstance(momentum, torch.Tensor):
                self.state.setdefault(param, {})["momentum_buffer"] = momentum.to(param.device)


def configure_adamw_optimizer(
    model: SpakieGPT,
    *,
    lr: float,
    weight_decay: float,
    runtime: RuntimeSettings,
) -> torch.optim.AdamW:
    decay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        **optimizer_kwargs(runtime),
    )


def configure_torch_optimizer(
    model: SpakieGPT,
    config: SpakieConfig,
    runtime: RuntimeSettings,
    *,
    kind: str,
    lr: float,
    weight_decay: float,
):
    optimizer_kind = normalize_optimizer_kind(kind)
    if optimizer_kind == "adamw":
        optimizer = configure_adamw_optimizer(model, lr=lr, weight_decay=weight_decay, runtime=runtime)
        optimizer.optimizer_kind = "adamw"  # type: ignore[attr-defined]
        return optimizer
    return MuonAdamW(
        model.named_parameters(),
        lr=lr,
        weight_decay=weight_decay,
        runtime=runtime,
        settings=muon_settings_from_config(config),
    )


def set_optimizer_lr(optimizer, lr: float) -> None:
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
        return
    for group in optimizer.param_groups:
        group["lr"] = lr
