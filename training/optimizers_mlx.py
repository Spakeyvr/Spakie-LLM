"""MLX optimizer builders for AdamW fallback and Muon-first training."""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from configs.default import SpakieConfig
from training.muon_core import (
    MuonSettings,
    adjusted_muon_lr,
    is_muon_parameter_name,
    is_qkv_parameter_name,
    muon_settings_from_config,
    normalize_optimizer_kind,
)


def _subset_tree_by_names(tree, names: set[str]):
    return tree_unflatten([(k, v) for k, v in tree_flatten(tree) if k in names and v is not None])


def _split_tree_by_ndim(tree, *, split_ndim: int = 2):
    flat = tree_flatten(tree)
    high = [(k, v) for k, v in flat if v is not None and len(v.shape) >= split_ndim]
    low = [(k, v) for k, v in flat if v is not None and len(v.shape) < split_ndim]
    return tree_unflatten(high), tree_unflatten(low)


def _flatten_arrays(tree) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(tree) if isinstance(v, mx.array)}


def muon_newton_schulz_mlx(
    update: mx.array,
    *,
    ns_steps: int = 5,
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> mx.array:
    if len(update.shape) != 2:
        raise ValueError("Muon Newton-Schulz expects a 2D array")
    a, b, c = ns_coefficients
    x = update.astype(mx.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (mx.sqrt((x * x).sum()) + eps)
    for _ in range(ns_steps):
        xx_t = x @ x.T
        poly = mx.addmm(b * xx_t, xx_t, xx_t, alpha=c, beta=1.0)
        x = mx.addmm(a * x, poly, x, alpha=1.0, beta=1.0)
    if transposed:
        x = x.T
    return x.astype(update.dtype)


class DualAdamW:
    """Two AdamW optimizers: one with weight decay on >=2D params, one without."""

    optimizer_kind = "adamw"

    def __init__(self, learning_rate: float, weight_decay: float, betas: tuple[float, float]):
        self._weight_decay = weight_decay
        self.decay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
        )
        self.nodecay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=0.0, betas=betas
        )

    @property
    def learning_rate(self):
        return self.decay.learning_rate

    def set_lr(self, lr: float) -> None:
        self.decay.learning_rate = lr
        self.nodecay.learning_rate = lr

    def update(self, model, grads) -> None:
        decay_grads, nodecay_grads = _split_tree_by_ndim(grads)
        self.decay.update(model, decay_grads)
        self.nodecay.update(model, nodecay_grads)

    def state_trees(self) -> dict:
        return {"decay": self.decay.state, "nodecay": self.nodecay.state}

    def load_state_trees(self, state: dict) -> None:
        self.decay.state = state["decay"]
        self.nodecay.state = state["nodecay"]

    def eval_state(self) -> None:
        mx.eval(self.decay.state, self.nodecay.state)


class MuonAdamWMLX:
    """Hybrid MLX optimizer: Muon for hidden matrices, AdamW for auxiliary params."""

    optimizer_kind = "muon"

    def __init__(
        self,
        model,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        settings: MuonSettings,
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.settings = settings
        self.muon_state: dict[str, mx.array] = {}
        param_flat = _flatten_arrays(model.parameters())
        self.muon_names = [
            name for name, param in param_flat.items() if is_muon_parameter_name(name, len(param.shape))
        ]
        self.aux_decay_names = [
            name for name, param in param_flat.items() if name not in self.muon_names and len(param.shape) >= 2
        ]
        self.aux_nodecay_names = [
            name for name, param in param_flat.items() if name not in self.muon_names and len(param.shape) < 2
        ]
        self.aux_decay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas
        )
        self.aux_nodecay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=0.0, betas=betas
        )

    def set_lr(self, lr: float) -> None:
        self.learning_rate = lr
        self.aux_decay.learning_rate = lr
        self.aux_nodecay.learning_rate = lr

    def update(self, model, grads) -> None:
        grad_flat = _flatten_arrays(grads)
        param_flat = _flatten_arrays(model.parameters())
        if self.aux_decay_names:
            decay_grads = _subset_tree_by_names(grads, set(self.aux_decay_names))
            self.aux_decay.update(model, decay_grads)
        if self.aux_nodecay_names:
            nodecay_grads = _subset_tree_by_names(grads, set(self.aux_nodecay_names))
            self.aux_nodecay.update(model, nodecay_grads)

        updates: dict[str, mx.array] = {}
        for name in self.muon_names:
            grad = grad_flat.get(name)
            param = param_flat.get(name)
            if grad is None or param is None:
                continue
            momentum = self.muon_state.get(name)
            if momentum is None:
                momentum = mx.zeros_like(param)
            momentum = momentum * self.settings.momentum + grad
            self.muon_state[name] = momentum
            update = grad + momentum * self.settings.momentum if self.settings.nesterov else momentum
            next_param = param
            if self.weight_decay:
                next_param = next_param * (1.0 - self.learning_rate * self.weight_decay)
            next_param = next_param - self._orthogonal_update(name, update, param.dtype)
            updates[name] = next_param
        if updates:
            model.update(tree_unflatten(list(updates.items())))

    def _orthogonal_update(self, name: str, update: mx.array, dtype) -> mx.array:
        if self.settings.qkv_split and is_qkv_parameter_name(name) and update.shape[0] % 3 == 0:
            pieces = []
            for chunk in mx.split(update, 3, axis=0):
                orthogonal = muon_newton_schulz_mlx(
                    chunk,
                    ns_steps=self.settings.ns_steps,
                    ns_coefficients=self.settings.ns_coefficients,
                    eps=self.settings.eps,
                ).astype(dtype)
                chunk_lr = adjusted_muon_lr(
                    self.learning_rate, tuple(chunk.shape), self.settings.adjust_lr_fn
                )
                pieces.append(orthogonal * chunk_lr)
            return mx.concatenate(pieces, axis=0)
        orthogonal = muon_newton_schulz_mlx(
            update,
            ns_steps=self.settings.ns_steps,
            ns_coefficients=self.settings.ns_coefficients,
            eps=self.settings.eps,
        ).astype(dtype)
        return orthogonal * adjusted_muon_lr(
            self.learning_rate, tuple(update.shape), self.settings.adjust_lr_fn
        )

    def state_trees(self) -> dict:
        return {
            "muon": tree_unflatten(list(self.muon_state.items())),
            "aux_decay": self.aux_decay.state,
            "aux_nodecay": self.aux_nodecay.state,
        }

    def load_state_trees(self, state: dict) -> None:
        self.muon_state = _flatten_arrays(state.get("muon", {}))
        self.aux_decay.state = state["aux_decay"]
        self.aux_nodecay.state = state["aux_nodecay"]

    def eval_state(self) -> None:
        mx.eval(
            tree_unflatten(list(self.muon_state.items())),
            self.aux_decay.state,
            self.aux_nodecay.state,
        )


def configure_mlx_optimizer(
    model,
    config: SpakieConfig,
    *,
    kind: str,
    learning_rate: float,
    weight_decay: float,
):
    optimizer_kind = normalize_optimizer_kind(kind)
    if optimizer_kind == "adamw":
        return DualAdamW(learning_rate=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95))
    return MuonAdamWMLX(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        settings=muon_settings_from_config(config),
    )
