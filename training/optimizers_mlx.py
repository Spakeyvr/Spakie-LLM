"""MLX optimizer builders for AdamW fallback and Muon-first training."""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from configs.default import SpakieConfig
from training.muon_core import (
    MuonPrecomputeError,
    MuonSettings,
    adjusted_muon_lr,
    is_muon_parameter_name,
    muon_projection_split_count,
    muon_settings_from_config,
    normalize_optimizer_kind,
)


def _subset_tree_by_names(tree, names: set[str]):
    return tree_unflatten([(k, v) for k, v in tree_flatten(tree) if k in names and v is not None])


def _flatten_arrays(tree) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(tree) if isinstance(v, mx.array)}


def _cast_floating_tree(tree, dtype=mx.float32):
    """Cast floating leaves while preserving integer optimizer metadata."""
    def cast(value):
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(dtype)
        return value

    return tree_map(cast, tree)


def _apply_optimizer_to_master(
    optimizer,
    master_params: dict[str, mx.array],
    grads: dict[str, mx.array],
) -> None:
    """Apply an MLX optimizer to FP32 master leaves and update them in place."""
    if not grads:
        return
    grad_tree = tree_unflatten(
        [(name, grad.astype(mx.float32)) for name, grad in grads.items()]
    )
    param_tree = tree_unflatten([(name, master_params[name]) for name in grads])
    updated = optimizer.apply_gradients(grad_tree, param_tree)
    for name, value in tree_flatten(updated):
        master_params[name] = value.astype(mx.float32)


def _publish_master_params(
    model,
    master_params: dict[str, mx.array],
    model_dtypes: dict[str, object],
) -> dict:
    published = tree_unflatten(
        [
            (name, value.astype(model_dtypes[name]))
            for name, value in master_params.items()
        ]
    )
    model.update(published)
    return published


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


def _muon_newton_schulz_stacked_mlx(
    update: mx.array,
    *,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
) -> mx.array:
    if len(update.shape) != 3:
        raise ValueError("stacked Muon Newton-Schulz expects a 3D array")
    a, b, c = ns_coefficients
    x = update.astype(mx.float32)
    transposed = x.shape[1] > x.shape[2]
    if transposed:
        x = x.transpose(0, 2, 1)
    norm = mx.sqrt((x * x).sum(axis=(1, 2), keepdims=True)) + eps
    x = x / norm
    for _ in range(ns_steps):
        xx_t = x @ x.transpose(0, 2, 1)
        poly = mx.addmm(b * xx_t, xx_t, xx_t, alpha=c, beta=1.0)
        x = mx.addmm(a * x, poly, x, alpha=1.0, beta=1.0)
    if transposed:
        x = x.transpose(0, 2, 1)
    return x.astype(update.dtype)


class DualAdamW:
    """Two AdamW optimizers: one with weight decay on >=2D params, one without."""

    optimizer_kind = "adamw"

    def __init__(self, model, learning_rate: float, weight_decay: float, betas: tuple[float, float]):
        self._weight_decay = weight_decay
        self.decay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas,
            bias_correction=True,
        )
        self.nodecay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=0.0, betas=betas,
            bias_correction=True,
        )
        self.model_parameters = model.parameters()
        param_flat = _flatten_arrays(self.model_parameters)
        self._model_dtypes = {name: value.dtype for name, value in param_flat.items()}
        self._decay_names = {
            name for name, value in param_flat.items() if len(value.shape) >= 2
        }
        self.master_params: dict[str, mx.array] = {
            name: value.astype(mx.float32) for name, value in param_flat.items()
        }

    @property
    def learning_rate(self):
        return self.decay.learning_rate

    def set_lr(self, lr: float) -> None:
        self.decay.learning_rate = lr
        self.nodecay.learning_rate = lr

    def update(self, model, grads) -> None:
        grad_flat = _flatten_arrays(grads)
        decay_grads = {
            name: grad for name, grad in grad_flat.items() if name in self._decay_names
        }
        nodecay_grads = {
            name: grad for name, grad in grad_flat.items() if name not in self._decay_names
        }
        _apply_optimizer_to_master(self.decay, self.master_params, decay_grads)
        _apply_optimizer_to_master(self.nodecay, self.master_params, nodecay_grads)
        self.model_parameters = _publish_master_params(
            model, self.master_params, self._model_dtypes
        )

    def state_trees(self) -> dict:
        return {
            "master": tree_unflatten(list(self.master_params.items())),
            "decay": self.decay.state,
            "nodecay": self.nodecay.state,
        }

    def load_state_trees(self, state: dict) -> None:
        self.decay.state = _cast_floating_tree(state["decay"])
        self.nodecay.state = _cast_floating_tree(state["nodecay"])
        if "master" not in state:
            raise ValueError("MLX optimizer checkpoint is missing FP32 master parameters")
        master = state["master"]
        self.master_params = {
            name: value.astype(mx.float32)
            for name, value in _flatten_arrays(master).items()
        }

    def evaluation_state(self) -> tuple:
        return (
            self.model_parameters,
            self.master_params,
            self.decay.state,
            self.nodecay.state,
        )

    def eval_state(self) -> None:
        mx.eval(*self.evaluation_state())


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
        grouped_muon: bool = False,
        compile_muon_ns: bool = False,
        muon_route: str = "all",
    ) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.settings = settings
        self.grouped_muon = grouped_muon
        self.compile_muon_ns = compile_muon_ns
        self._compiled_muon_ns = None
        self._compiled_stacked_muon_ns: dict[tuple[tuple[int, ...], str], object] = {}
        if self.compile_muon_ns:
            ns_steps = self.settings.ns_steps
            ns_coefficients = self.settings.ns_coefficients
            eps = self.settings.eps

            @mx.compile
            def _compiled_muon_ns(update: mx.array) -> mx.array:
                return muon_newton_schulz_mlx(
                    update,
                    ns_steps=ns_steps,
                    ns_coefficients=ns_coefficients,
                    eps=eps,
                )

            self._compiled_muon_ns = _compiled_muon_ns
        self.muon_state: dict[str, mx.array] = {}
        self.model_parameters = model.parameters()
        param_flat = _flatten_arrays(self.model_parameters)
        self._model_dtypes = {name: value.dtype for name, value in param_flat.items()}
        muon_route = (muon_route or "all").lower()

        def _route_allows(name: str) -> bool:
            if muon_route == "all":
                return True
            if muon_route == "mlp":
                return ".mlp." in name
            if muon_route == "attn":
                return ".attn." in name
            if muon_route == "none":
                return False
            return True

        self.muon_names = [
            name
            for name, param in param_flat.items()
            if is_muon_parameter_name(name, len(param.shape)) and _route_allows(name)
        ]
        self.aux_decay_names = [
            name for name, param in param_flat.items() if name not in self.muon_names and len(param.shape) >= 2
        ]
        self.aux_nodecay_names = [
            name for name, param in param_flat.items() if name not in self.muon_names and len(param.shape) < 2
        ]
        self.aux_decay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay, betas=betas,
            bias_correction=True,
        )
        self.aux_nodecay = optim.AdamW(
            learning_rate=learning_rate, weight_decay=0.0, betas=betas,
            bias_correction=True,
        )
        self.master_params: dict[str, mx.array] = {
            name: value.astype(mx.float32) for name, value in param_flat.items()
        }

    def set_lr(self, lr: float) -> None:
        self.learning_rate = lr
        self.aux_decay.learning_rate = lr
        self.aux_nodecay.learning_rate = lr

    def update(self, model, grads) -> None:
        grad_flat = _flatten_arrays(grads)
        # Gradients from a BF16 forward pass are promoted before any optimizer
        # state or master-parameter arithmetic. This prevents Adam moments and
        # Muon momentum from inheriting BF16's coarse mantissa.
        grad_flat = {
            name: value.astype(mx.float32) for name, value in grad_flat.items()
        }
        param_flat = self.master_params
        try:
            if self.grouped_muon:
                updates, next_muon_state = self._prepare_muon_grouped(grad_flat, param_flat)
            else:
                updates, next_muon_state = self._prepare_muon(grad_flat, param_flat)

            # Materialize all potentially-failing Muon math before AdamW or
            # model state is mutated.
            mx.eval(
                tree_unflatten(list(updates.items())),
                tree_unflatten(list(next_muon_state.items())),
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            raise MuonPrecomputeError(f"Muon update preparation failed: {exc}") from exc
        if self.aux_decay_names:
            decay_grads = {
                name: grad_flat[name]
                for name in self.aux_decay_names
                if name in grad_flat
            }
            _apply_optimizer_to_master(self.aux_decay, self.master_params, decay_grads)
        if self.aux_nodecay_names:
            nodecay_grads = {
                name: grad_flat[name]
                for name in self.aux_nodecay_names
                if name in grad_flat
            }
            _apply_optimizer_to_master(
                self.aux_nodecay, self.master_params, nodecay_grads
            )
        # Muon updates are already expressed in FP32 master space. Publish a
        # cast view to the model for the next forward pass while retaining the
        # higher-precision values for the next optimizer step.
        if updates:
            self.master_params.update(updates)
        self.model_parameters = _publish_master_params(
            model, self.master_params, self._model_dtypes
        )
        self.muon_state = next_muon_state

    def _prepare_muon(
        self, grad_flat: dict[str, mx.array], param_flat: dict[str, mx.array]
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        next_muon_state = dict(self.muon_state)

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
            next_muon_state[name] = momentum
            update = grad + momentum * self.settings.momentum if self.settings.nesterov else momentum
            next_param = param
            if self.weight_decay:
                next_param = next_param * (1.0 - self.learning_rate * self.weight_decay)
            next_param = next_param - self._orthogonal_update(name, update, param.dtype)
            updates[name] = next_param
        return updates, next_muon_state

    def _prepare_muon_grouped(
        self, grad_flat: dict[str, mx.array], param_flat: dict[str, mx.array]
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        bases: dict[str, mx.array] = {}
        next_muon_state = dict(self.muon_state)
        grouped: dict[tuple[tuple[int, int], str, float], list[tuple[str, int | None, mx.array, object]]] = {}
        for name in self.muon_names:
            grad = grad_flat.get(name)
            param = param_flat.get(name)
            if grad is None or param is None:
                continue
            momentum = self.muon_state.get(name)
            if momentum is None:
                momentum = mx.zeros_like(param)
            momentum = momentum * self.settings.momentum + grad
            next_muon_state[name] = momentum
            update = grad + momentum * self.settings.momentum if self.settings.nesterov else momentum
            next_param = param
            if self.weight_decay:
                next_param = next_param * (1.0 - self.learning_rate * self.weight_decay)
            bases[name] = next_param

            split_count = muon_projection_split_count(name) if self.settings.qkv_split else 1
            if split_count > 1 and update.shape[0] % split_count == 0:
                for chunk_idx, chunk in enumerate(mx.split(update, split_count, axis=0)):
                    lr = adjusted_muon_lr(
                        self.learning_rate, tuple(chunk.shape), self.settings.adjust_lr_fn
                    )
                    key = (tuple(chunk.shape), str(param.dtype), float(lr))
                    grouped.setdefault(key, []).append((name, chunk_idx, chunk, param.dtype))
            else:
                lr = adjusted_muon_lr(
                    self.learning_rate, tuple(update.shape), self.settings.adjust_lr_fn
                )
                key = (tuple(update.shape), str(param.dtype), float(lr))
                grouped.setdefault(key, []).append((name, None, update, param.dtype))

        updates: dict[str, mx.array] = {}
        split_updates: dict[str, list[mx.array | None]] = {}
        for (_, _, lr), records in grouped.items():
            stacked = mx.stack([record[2] for record in records])
            orthogonal = self._newton_schulz_stacked(stacked)
            for idx, (name, chunk_idx, _, dtype) in enumerate(records):
                update_piece = orthogonal[idx].astype(dtype) * lr
                if chunk_idx is None:
                    updates[name] = bases[name] - update_piece
                else:
                    piece_count = muon_projection_split_count(name)
                    pieces = split_updates.setdefault(name, [None] * piece_count)
                    pieces[chunk_idx] = update_piece

        for name, pieces in split_updates.items():
            if any(piece is None for piece in pieces):
                raise RuntimeError(f"incomplete grouped projection update for {name}")
            updates[name] = bases[name] - mx.concatenate(pieces, axis=0)

        return updates, next_muon_state

    def _orthogonal_update(self, name: str, update: mx.array, dtype) -> mx.array:
        split_count = muon_projection_split_count(name) if self.settings.qkv_split else 1
        if split_count > 1 and update.shape[0] % split_count == 0:
            pieces = []
            for chunk in mx.split(update, split_count, axis=0):
                orthogonal = self._newton_schulz(chunk).astype(dtype)
                chunk_lr = adjusted_muon_lr(
                    self.learning_rate, tuple(chunk.shape), self.settings.adjust_lr_fn
                )
                pieces.append(orthogonal * chunk_lr)
            return mx.concatenate(pieces, axis=0)
        orthogonal = self._newton_schulz(update).astype(dtype)
        return orthogonal * adjusted_muon_lr(
            self.learning_rate, tuple(update.shape), self.settings.adjust_lr_fn
        )

    def _newton_schulz(self, update: mx.array) -> mx.array:
        if self._compiled_muon_ns is not None:
            return self._compiled_muon_ns(update)
        return muon_newton_schulz_mlx(
            update,
            ns_steps=self.settings.ns_steps,
            ns_coefficients=self.settings.ns_coefficients,
            eps=self.settings.eps,
        )

    def _newton_schulz_stacked(self, update: mx.array) -> mx.array:
        if self.compile_muon_ns:
            key = (tuple(int(dim) for dim in update.shape), str(update.dtype))
            compiled = self._compiled_stacked_muon_ns.get(key)
            if compiled is None:
                ns_steps = self.settings.ns_steps
                ns_coefficients = self.settings.ns_coefficients
                eps = self.settings.eps

                @mx.compile
                def _compiled_stacked_muon_ns(stacked_update: mx.array) -> mx.array:
                    return _muon_newton_schulz_stacked_mlx(
                        stacked_update,
                        ns_steps=ns_steps,
                        ns_coefficients=ns_coefficients,
                        eps=eps,
                    )

                compiled = _compiled_stacked_muon_ns
                self._compiled_stacked_muon_ns[key] = compiled
            return compiled(update)
        return _muon_newton_schulz_stacked_mlx(
            update,
            ns_steps=self.settings.ns_steps,
            ns_coefficients=self.settings.ns_coefficients,
            eps=self.settings.eps,
        )

    def state_trees(self) -> dict:
        return {
            "master": tree_unflatten(list(self.master_params.items())),
            "muon": tree_unflatten(list(self.muon_state.items())),
            "aux_decay": self.aux_decay.state,
            "aux_nodecay": self.aux_nodecay.state,
        }

    def load_state_trees(self, state: dict) -> None:
        if "master" not in state:
            raise ValueError("MLX optimizer checkpoint is missing FP32 master parameters")
        self.master_params = {
            name: value.astype(mx.float32)
            for name, value in _flatten_arrays(state["master"]).items()
        }
        self.muon_state = {
            name: value.astype(mx.float32)
            for name, value in _flatten_arrays(state.get("muon", {})).items()
        }
        self.aux_decay.state = _cast_floating_tree(state["aux_decay"])
        self.aux_nodecay.state = _cast_floating_tree(state["aux_nodecay"])

    def evaluation_state(self) -> tuple:
        return (
            self.model_parameters,
            self.master_params,
            self.muon_state,
            self.aux_decay.state,
            self.aux_nodecay.state,
        )

    def eval_state(self) -> None:
        mx.eval(*self.evaluation_state())


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
        return DualAdamW(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
    return MuonAdamWMLX(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        settings=muon_settings_from_config(config),
        grouped_muon=bool(getattr(config, "grouped_muon", False)),
        compile_muon_ns=bool(getattr(config, "compile_muon_ns", False)),
        muon_route=str(getattr(config, "muon_route", "all")),
    )
