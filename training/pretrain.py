"""Pretraining loop with cosine LR, gradient accumulation, and token budgets."""

import atexit
import math
import os
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION, SpakieConfig, config_to_dict
from model.transformer import SpakieGPT
from model.utils import count_parameters
from runtime import RuntimeSettings, autocast_context
from runtime.backends import create_grad_scaler
from runtime.checkpoint_io import atomic_torch_save
from training.monitor import (
    TrainingStatusWriter,
    format_monitor_start_message,
    start_background_monitor,
    stop_background_monitor,
)
from training.muon_core import adamw_fallback_warning, should_adamw_fallback
from training.optimizers import configure_torch_optimizer, set_optimizer_lr


class ResumableBatchSampler:
    """Deterministic shuffled sampler that can resume at an exact batch boundary."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        drop_last: bool = True,
        generator_state=None,
        indices: torch.Tensor | None = None,
        position: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = torch.Generator()
        if generator_state is not None:
            self.generator.set_state(generator_state)
        self.indices = indices.clone() if indices is not None else torch.empty(0, dtype=torch.int64)
        self.position = position
        if self.indices.numel() == 0:
            self._refresh_indices()

    def _refresh_indices(self) -> None:
        self.indices = torch.randperm(self.dataset_size, generator=self.generator, dtype=torch.int64)
        self.position = 0

    def __iter__(self):
        while True:
            remaining = self.indices.numel() - self.position
            if remaining < self.batch_size:
                if not self.drop_last and remaining > 0:
                    batch = self.indices[self.position :].tolist()
                    self._refresh_indices()
                    yield batch
                else:
                    self._refresh_indices()
                continue

            next_position = self.position + self.batch_size
            batch = self.indices[self.position : next_position].tolist()
            self.position = next_position
            yield batch

    def state_dict(self) -> dict[str, object]:
        return {
            "generator_state": self.generator.get_state(),
            "indices": self.indices.clone(),
            "position": self.position,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]):
        return cls(
            dataset_size=int(state["dataset_size"]),
            batch_size=int(state["batch_size"]),
            drop_last=bool(state.get("drop_last", True)),
            generator_state=state.get("generator_state"),
            indices=state.get("indices"),
            position=int(state.get("position", 0)),
        )


def capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, dict):
            keys = numpy_state["keys"]
            if isinstance(keys, torch.Tensor):
                keys = keys.cpu().numpy().astype(np.uint32, copy=False)
            np.random.set_state((
                numpy_state["bit_generator"],
                keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            ))
        else:
            # Unsafe legacy checkpoints used NumPy's native tuple representation.
            np.random.set_state(numpy_state)
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str,
    *,
    model: SpakieGPT,
    optimizer: torch.optim.Optimizer,
    config: SpakieConfig,
    global_step: int,
    tokens_processed: int,
    best_val_loss: float,
    val_loss: float | None,
    elapsed_time: float,
    train_sampler: ResumableBatchSampler,
    scaler=None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_kind": getattr(optimizer, "optimizer_kind", config.pretrain_optimizer),
        "optimizer_warning": "fallback_not_recommended"
        if getattr(optimizer, "optimizer_kind", config.pretrain_optimizer) == "adamw"
        else "",
        "muon_hyperparameters": {
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_steps": config.muon_ns_steps,
            "ns_coefficients": list(config.muon_ns_coefficients),
            "eps": config.muon_eps,
            "adjust_lr_fn": config.muon_adjust_lr_fn,
            "qkv_split": config.muon_qkv_split,
        },
        "muon_verified": config.muon_verified,
        "step": global_step,
        "tokens_processed": tokens_processed,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "elapsed_time": elapsed_time,
        "train_sampler": train_sampler.state_dict(),
        "rng_state": capture_rng_state(),
        "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
        "config": config_to_dict(config),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    atomic_torch_save(payload, path)


def get_lr(step: int, config: SpakieConfig) -> float:
    """Learning rate at `step`. See training/pretrain_mlx.py:get_lr for the
    schedule semantics (cosine | trapezoid). Must stay numerically identical
    to the MLX path so backends share a learning curve.
    """
    min_lr = config.pretrain_lr * 0.1
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / max(config.pretrain_warmup_steps, 1)
    if step >= config.pretrain_max_steps:
        return min_lr
    schedule = getattr(config, "pretrain_lr_schedule", "cosine")
    if schedule == "trapezoid":
        decay_frac = max(0.0, min(1.0, getattr(config, "pretrain_trapezoid_decay_frac", 0.2)))
        decay_start = int(config.pretrain_max_steps * (1.0 - decay_frac))
        if step < decay_start:
            return config.pretrain_lr
        progress = (step - decay_start) / max(config.pretrain_max_steps - decay_start, 1)
        return config.pretrain_lr - progress * (config.pretrain_lr - min_lr)
    progress = (step - config.pretrain_warmup_steps) / (config.pretrain_max_steps - config.pretrain_warmup_steps)
    return min_lr + 0.5 * (config.pretrain_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: SpakieGPT, val_loader: DataLoader, config: SpakieConfig, runtime: RuntimeSettings) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= config.pretrain_eval_batches:
            break
        x, y = x.to(runtime.device), y.to(runtime.device)
        with autocast_context(runtime):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def pretrain(model: SpakieGPT, train_loader: DataLoader, val_loader: DataLoader,
             config: SpakieConfig, runtime: RuntimeSettings,
             resume_state: dict[str, object] | None = None,
             allow_adamw_fallback: bool = False):
    model.to(runtime.device)
    model.train()
    optimizer = configure_torch_optimizer(
        model,
        config,
        runtime,
        kind=config.pretrain_optimizer,
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
    )
    scaler = create_grad_scaler(runtime)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    train_sampler = getattr(train_loader, "batch_sampler", None)
    if not isinstance(train_sampler, ResumableBatchSampler):
        raise TypeError("Pretraining requires a ResumableBatchSampler for exact resume support.")
    # DataLoader workers can request batches ahead of the training loop. This
    # independent cursor advances only after an optimizer step commits, so a
    # checkpoint never skips prefetched-but-untrained examples.
    committed_sampler = ResumableBatchSampler.from_state_dict(train_sampler.state_dict())
    committed_iter = iter(committed_sampler)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    tokens_processed = 0
    target_tokens = config.pretrain_target_tokens
    elapsed_before_resume = 0.0
    interrupted = False
    last_val_loss = None
    stop_requested = False

    if resume_state:
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        best_val_loss = float(resume_state.get("best_val_loss", resume_state.get("val_loss", best_val_loss)))
        global_step = int(resume_state.get("step", 0))
        tokens_processed = int(resume_state.get("tokens_processed", 0))
        elapsed_before_resume = float(resume_state.get("elapsed_time", 0.0))
        last_val_loss = resume_state.get("val_loss")
        restore_rng_state(resume_state.get("rng_state"))
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])

    start_time = time.time() - elapsed_before_resume
    status_writer = TrainingStatusWriter(
        config.checkpoint_dir,
        stage="pretrain",
        backend="torch",
        preset=config.preset_name,
        total_steps=config.pretrain_max_steps,
        target_tokens=target_tokens,
        elapsed_offset=elapsed_before_resume,
    )
    status_writer.update(
        force=True,
        status="running",
        step=global_step,
        tokens_processed=tokens_processed,
        best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        val_loss=last_val_loss,
    )
    monitor_info = start_background_monitor(status_writer.path, config.checkpoint_dir)
    print(format_monitor_start_message(monitor_info))
    atexit.register(stop_background_monitor, monitor_info)
    train_iter = iter(train_loader)
    pbar = tqdm(total=config.pretrain_max_steps, desc="Pretrain", initial=global_step)

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        pbar.write("\nStop requested. Finishing the current optimizer step before saving resume state...")

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while global_step < config.pretrain_max_steps and (target_tokens <= 0 or tokens_processed < target_tokens):
            optimizer.zero_grad(set_to_none=True)
            accum_loss_tensor = None
            consumed_microbatches = 0
            step_tokens = 0

            for micro_step in range(config.pretrain_grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(runtime.device), y.to(runtime.device)
                with autocast_context(runtime):
                    _, loss = model(x, y)
                    loss = loss / config.pretrain_grad_accum_steps

                scaler.scale(loss).backward()
                detached = loss.detach()
                accum_loss_tensor = detached if accum_loss_tensor is None else accum_loss_tensor + detached
                step_tokens += x.numel()
                consumed_microbatches += 1

            accum_loss = accum_loss_tensor.item()  # single GPU→CPU sync per optimizer step

            scaler.unscale_(optimizer)
            if config.pretrain_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.pretrain_grad_clip)

            # Update LR
            lr = get_lr(global_step, config)
            set_optimizer_lr(optimizer, lr)

            # MPS bug: addcmul_/addcdiv_ silently fails on non-contiguous tensors (AdamW internals),
            # causing weights to stop updating. Force contiguous before the step.
            if runtime.device.type == "mps":
                for p in model.parameters():
                    if p.grad is not None and not p.grad.is_contiguous():
                        p.grad = p.grad.contiguous()

            try:
                scaler.step(optimizer)
            except Exception as exc:
                if should_adamw_fallback(
                    exc, optimizer, config, stage="pretrain", allow=allow_adamw_fallback
                ):
                    print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
                    config.pretrain_optimizer = "adamw"
                    print(adamw_fallback_warning("Pretraining"))
                    optimizer = configure_torch_optimizer(
                        model,
                        config,
                        runtime,
                        kind="adamw",
                        lr=config.pretrain_lr,
                        weight_decay=config.pretrain_weight_decay,
                    )
                    set_optimizer_lr(optimizer, lr)
                    # Gradients were already unscaled against the failed Muon
                    # optimizer, so stepping the replacement directly avoids
                    # applying the inverse scale twice.
                    optimizer.step()
                else:
                    raise
            scaler.update()
            global_step += 1
            tokens_processed += step_tokens
            for _ in range(consumed_microbatches):
                next(committed_iter)
            pbar.update(1)
            elapsed = time.time() - start_time
            status_writer.update(
                status="running",
                step=global_step,
                total_steps=config.pretrain_max_steps,
                train_loss=accum_loss,
                val_loss=last_val_loss,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                lr=lr,
                tokens_processed=tokens_processed,
                target_tokens=target_tokens,
                tok_per_sec=tokens_processed / elapsed if elapsed > 0 else 0.0,
            )

            if stop_requested:
                interrupted = True
                interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.pt")
                save_training_checkpoint(
                    interrupt_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    tokens_processed=tokens_processed,
                    best_val_loss=best_val_loss,
                    val_loss=last_val_loss,
                    elapsed_time=time.time() - start_time,
                    train_sampler=committed_sampler,
                    scaler=scaler,
                )
                status_writer.finish(
                    "interrupted",
                    message=f"Saved checkpoint to {interrupt_path}",
                    step=global_step,
                    tokens_processed=tokens_processed,
                    val_loss=last_val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                )
                pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
                break

            checkpoint_interval = int(getattr(config, "pretrain_checkpoint_interval", 0) or 0)
            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                rolling_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.pt")
                save_training_checkpoint(
                    rolling_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    global_step=global_step,
                    tokens_processed=tokens_processed,
                    best_val_loss=best_val_loss,
                    val_loss=last_val_loss,
                    elapsed_time=time.time() - start_time,
                    train_sampler=committed_sampler,
                    scaler=scaler,
                )
                status_writer.update(force=True, last_checkpoint=rolling_path)
                pbar.write(f"  -> saved rolling checkpoint at step {global_step}")

            # Eval
            if global_step % config.pretrain_eval_interval == 0:
                val_loss = evaluate(model, val_loader, config, runtime)
                last_val_loss = val_loss
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed
                token_progress = ""
                if target_tokens > 0:
                    progress_pct = min(tokens_processed / target_tokens * 100.0, 100.0)
                    token_progress = f" | tokens {tokens_processed:,}/{target_tokens:,} ({progress_pct:.1f}%)"
                pbar.write(
                    f"step {global_step} | train_loss {accum_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_per_sec:.0f}{token_progress}"
                )
                status_writer.update(
                    force=True,
                    status="running",
                    step=global_step,
                    train_loss=accum_loss,
                    val_loss=val_loss,
                    best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                    lr=lr,
                    tokens_processed=tokens_processed,
                    target_tokens=target_tokens,
                    tok_per_sec=tok_per_sec,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = os.path.join(config.checkpoint_dir, "pretrain_best.pt")
                    save_training_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        global_step=global_step,
                        tokens_processed=tokens_processed,
                        best_val_loss=best_val_loss,
                        val_loss=val_loss,
                        elapsed_time=elapsed,
                        train_sampler=committed_sampler,
                        scaler=scaler,
                    )
                    status_writer.update(
                        force=True,
                        best_val_loss=best_val_loss,
                        best_checkpoint=ckpt_path,
                    )
                    pbar.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")
                elif config.should_use_pretrain_early_stopping():
                    patience_counter += 1
                    if patience_counter >= config.pretrain_patience:
                        pbar.write(f"Early stopping at step {global_step}")
                        status_writer.finish(
                            "stopped",
                            message=f"Early stopping at step {global_step}",
                            step=global_step,
                            train_loss=accum_loss,
                            val_loss=last_val_loss,
                            best_val_loss=best_val_loss,
                            tokens_processed=tokens_processed,
                        )
                        break
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.pt")
        save_training_checkpoint(
            interrupt_path,
            model=model,
            optimizer=optimizer,
            config=config,
            global_step=global_step,
            tokens_processed=tokens_processed,
            best_val_loss=best_val_loss,
            val_loss=last_val_loss,
            elapsed_time=time.time() - start_time,
            train_sampler=committed_sampler,
            scaler=scaler,
        )
        status_writer.finish(
            "interrupted",
            message=f"Saved checkpoint to {interrupt_path}",
            step=global_step,
            tokens_processed=tokens_processed,
            val_loss=last_val_loss,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        )
        pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)
        pbar.close()

    if interrupted:
        print(f"Pretraining stopped early at {tokens_processed:,} tokens. Best val loss so far: {best_val_loss:.4f}")
    else:
        final_path = os.path.join(config.checkpoint_dir, "pretrain_final.pt")
        save_training_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            config=config,
            global_step=global_step,
            tokens_processed=tokens_processed,
            best_val_loss=best_val_loss,
            val_loss=last_val_loss,
            elapsed_time=time.time() - start_time,
            train_sampler=committed_sampler,
            scaler=scaler,
        )
        if status_writer.payload.get("status") != "stopped":
            status_writer.finish(
                "complete",
                message="Pretraining complete",
                step=global_step,
                tokens_processed=tokens_processed,
                val_loss=last_val_loss,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            )
        print(f"Pretraining complete at {tokens_processed:,} tokens. Best val loss: {best_val_loss:.4f}")
        print(f"Final checkpoint: {final_path}")
    if stop_background_monitor(monitor_info):
        print("Monitor stopped.")
    return best_val_loss
