"""SFT fine-tuning loop."""

import atexit
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION, SpakieConfig, config_to_dict
from model.transformer import SpakieGPT
from runtime import RuntimeSettings, autocast_context, dataloader_kwargs
from runtime.checkpoint_io import atomic_torch_save, checkpoint_tokenizer_contract
from runtime.backends import create_grad_scaler
from training.dataset import ChatSFTDataset, train_val_split
from training.monitor import (
    TrainingStatusWriter,
    format_monitor_start_message,
    start_background_monitor,
    stop_background_monitor,
)
from training.muon_core import adamw_fallback_warning, should_adamw_fallback
from training.optimizers import configure_torch_optimizer, set_optimizer_lr


def _parameter_iter(model_or_parameters):
    if hasattr(model_or_parameters, "parameters"):
        return model_or_parameters.parameters()
    return iter(model_or_parameters)


def _ensure_contiguous_mps_grads(model_or_parameters, runtime: RuntimeSettings) -> None:
    if runtime.device.type != "mps":
        return
    for param in _parameter_iter(model_or_parameters):
        if param.grad is not None and not param.grad.is_contiguous():
            param.grad = param.grad.contiguous()


def _scale_partial_sft_grads(model_or_parameters, config: SpakieConfig, microbatches_in_step: int) -> None:
    if microbatches_in_step <= 0:
        raise ValueError("microbatches_in_step must be positive")
    if microbatches_in_step == config.sft_grad_accum_steps:
        return
    scale = config.sft_grad_accum_steps / microbatches_in_step
    for param in _parameter_iter(model_or_parameters):
        if param.grad is not None:
            param.grad.mul_(scale)


def _consume_logged_losses(losses, total: float, scale: int) -> tuple[float, int]:
    """Materialize a group of loss scalars with one accelerator synchronization."""
    if not losses:
        return total, 0
    values = torch.stack(losses).tolist()
    for value in values:
        total += value * scale
    losses.clear()
    return total, len(values)


def _consume_weighted_losses(
    losses,
    token_counts,
    total_loss: float,
    supervised_tokens: int,
) -> tuple[float, int]:
    if not losses:
        return total_loss, supervised_tokens
    for value, token_count in zip(torch.stack(losses).tolist(), token_counts):
        total_loss += value * token_count
        supervised_tokens += token_count
    losses.clear()
    token_counts.clear()
    return total_loss, supervised_tokens


@torch.no_grad()
def _evaluate_sft_loss(model, val_loader, runtime: RuntimeSettings) -> float:
    """Return mean cross-entropy per supervised token, matching the MLX path."""
    model.eval()
    total_loss = 0.0
    supervised_tokens = 0
    losses = []
    token_counts = []
    non_blocking = runtime.device.type == "cuda"
    for x, y in val_loader:
        token_count = int((y != -100).sum().item())
        if token_count == 0:
            continue
        x = x.to(runtime.device, non_blocking=non_blocking)
        y = y.to(runtime.device, non_blocking=non_blocking)
        with autocast_context(runtime):
            _, loss = model(x, y)
        losses.append(loss.detach())
        token_counts.append(token_count)
        if len(losses) == 32:
            total_loss, supervised_tokens = _consume_weighted_losses(
                losses,
                token_counts,
                total_loss,
                supervised_tokens,
            )
    total_loss, supervised_tokens = _consume_weighted_losses(
        losses,
        token_counts,
        total_loss,
        supervised_tokens,
    )
    return total_loss / max(supervised_tokens, 1)


def _sft_optimizer_step(
    model: SpakieGPT,
    optimizer,
    config: SpakieConfig,
    runtime: RuntimeSettings,
    *,
    global_step: int,
    total_steps: int,
    microbatches_in_step: int,
    allow_adamw_fallback: bool,
    scaler: torch.amp.GradScaler,
    parameters=None,
) -> object:
    if parameters is None:
        parameters = tuple(model.parameters())
    scaler.unscale_(optimizer)
    _scale_partial_sft_grads(parameters, config, microbatches_in_step)
    if config.sft_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(parameters, config.sft_grad_clip)

    progress = global_step / max(total_steps, 1)
    lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (1 + math.cos(math.pi * progress))
    set_optimizer_lr(optimizer, lr)
    _ensure_contiguous_mps_grads(parameters, runtime)

    try:
        scaler.step(optimizer)
    except Exception as exc:
        if should_adamw_fallback(exc, optimizer, config, stage="sft", allow=allow_adamw_fallback):
            print(f"USING ADAMW FALLBACK after Muon failure: {exc}")
            config.sft_optimizer = "adamw"
            print(adamw_fallback_warning("SFT"))
            optimizer = configure_torch_optimizer(
                model,
                config,
                runtime,
                kind="adamw",
                lr=config.sft_lr,
                weight_decay=config.sft_weight_decay,
            )
            set_optimizer_lr(optimizer, lr)
            _ensure_contiguous_mps_grads(parameters, runtime)
            optimizer.step()
        else:
            raise
    scaler.update()
    return optimizer


def finetune(model: SpakieGPT, train_dataset: ChatSFTDataset, val_dataset,
             config: SpakieConfig, runtime: RuntimeSettings,
             num_workers: int = 2,
             best_checkpoint_name: str = "sft_best.pt",
             interrupt_checkpoint_name: str = "sft_interrupt.pt",
             allow_adamw_fallback: bool = False):
    model.to(runtime.device)
    model.train()

    loader_options = dataloader_kwargs(runtime, num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.sft_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.sft_batch_size,
        shuffle=False,
        **loader_options,
    )

    optimizer = configure_torch_optimizer(
        model,
        config,
        runtime,
        kind=config.sft_optimizer,
        lr=config.sft_lr,
        weight_decay=config.sft_weight_decay,
    )
    scaler = create_grad_scaler(runtime)
    parameters = tuple(model.parameters())
    non_blocking = runtime.device.type == "cuda"

    steps_per_epoch = math.ceil(len(train_loader) / config.sft_grad_accum_steps)
    total_steps = steps_per_epoch * config.sft_epochs
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False
    status_writer = TrainingStatusWriter(
        config.checkpoint_dir,
        stage="sft",
        backend="torch",
        preset=config.preset_name,
        total_steps=total_steps,
    )
    status_writer.update(force=True, status="running")
    monitor_info = start_background_monitor(status_writer.path, config.checkpoint_dir)
    print(format_monitor_start_message(monitor_info))
    atexit.register(stop_background_monitor, monitor_info)

    try:
        for epoch in range(config.sft_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pending_grads = False
            pending_microbatches = 0
            pending_losses = []

            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")
            try:
                for batch_idx, (x, y) in enumerate(pbar):
                    x = x.to(runtime.device, non_blocking=non_blocking)
                    y = y.to(runtime.device, non_blocking=non_blocking)
                    with autocast_context(runtime):
                        _, loss = model(x, y)
                        loss = loss / config.sft_grad_accum_steps

                    scaler.scale(loss).backward()
                    pending_grads = True
                    pending_microbatches += 1
                    pending_losses.append(loss.detach())

                    if (batch_idx + 1) % config.sft_grad_accum_steps == 0:
                        epoch_loss, recorded = _consume_logged_losses(
                            pending_losses,
                            epoch_loss,
                            config.sft_grad_accum_steps,
                        )
                        n_batches += recorded
                        optimizer = _sft_optimizer_step(
                            model,
                            optimizer,
                            config,
                            runtime,
                            global_step=global_step,
                            total_steps=total_steps,
                            microbatches_in_step=pending_microbatches,
                            allow_adamw_fallback=allow_adamw_fallback,
                            scaler=scaler,
                            parameters=parameters,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        pending_grads = False
                        pending_microbatches = 0
                        global_step += 1
                        status_writer.update(
                            status="running",
                            epoch=epoch + 1,
                            epochs=config.sft_epochs,
                            step=global_step,
                            total_steps=total_steps,
                            train_loss=epoch_loss / max(n_batches, 1),
                        )
                        pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")
            finally:
                pbar.close()

            if pending_grads:
                epoch_loss, recorded = _consume_logged_losses(
                    pending_losses,
                    epoch_loss,
                    config.sft_grad_accum_steps,
                )
                n_batches += recorded
                optimizer = _sft_optimizer_step(
                    model,
                    optimizer,
                    config,
                    runtime,
                    global_step=global_step,
                    total_steps=total_steps,
                    microbatches_in_step=pending_microbatches,
                    allow_adamw_fallback=allow_adamw_fallback,
                    scaler=scaler,
                    parameters=parameters,
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                status_writer.update(
                    status="running",
                    epoch=epoch + 1,
                    epochs=config.sft_epochs,
                    step=global_step,
                    total_steps=total_steps,
                    train_loss=epoch_loss / max(n_batches, 1),
                )

            # Validation
            val_loss = _evaluate_sft_loss(model, val_loader, runtime)
            print(f"Epoch {epoch + 1} | train_loss {epoch_loss / n_batches:.4f} | val_loss {val_loss:.4f}")
            status_writer.update(
                force=True,
                status="running",
                epoch=epoch + 1,
                epochs=config.sft_epochs,
                step=global_step,
                total_steps=total_steps,
                train_loss=epoch_loss / max(n_batches, 1),
                val_loss=val_loss,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(config.checkpoint_dir, best_checkpoint_name)
                atomic_torch_save({
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                    "optimizer_warning": "fallback_not_recommended"
                    if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                    else "",
                    "muon_verified": config.muon_verified,
                    "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
                    "config": config_to_dict(config),
                    "tokenizer": checkpoint_tokenizer_contract(config),
                }, ckpt_path)
                status_writer.update(
                    force=True,
                    best_val_loss=best_val_loss,
                    best_checkpoint=ckpt_path,
                )
                print(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.sft_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    status_writer.finish(
                        "stopped",
                        message=f"Early stopping at epoch {epoch + 1}",
                        epoch=epoch + 1,
                        epochs=config.sft_epochs,
                        step=global_step,
                        total_steps=total_steps,
                        val_loss=val_loss,
                        best_val_loss=best_val_loss,
                    )
                    break
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, interrupt_checkpoint_name)
        atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
            "optimizer_warning": "fallback_not_recommended"
            if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
            else "",
            "muon_verified": config.muon_verified,
            "step": global_step,
            "best_val_loss": best_val_loss,
            "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
            "config": config_to_dict(config),
            "tokenizer": checkpoint_tokenizer_contract(config),
        }, interrupt_path)
        status_writer.finish(
            "interrupted",
            message=f"Saved checkpoint to {interrupt_path}",
            step=global_step,
            total_steps=total_steps,
            best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        )
        print(f"\nInterrupted during fine-tuning. Saved checkpoint to {interrupt_path}")

    if interrupted:
        print(f"SFT stopped early. Best val loss so far: {best_val_loss:.4f}")
    else:
        if status_writer.payload.get("status") != "stopped":
            status_writer.finish(
                "complete",
                message="SFT complete",
                step=global_step,
                total_steps=total_steps,
                best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
            )
        print(f"SFT complete. Best val loss: {best_val_loss:.4f}")
    if stop_background_monitor(monitor_info):
        print("Monitor stopped.")
    return best_val_loss
