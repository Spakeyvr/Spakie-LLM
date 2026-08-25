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
from training.pretrain import (
    ResumableBatchSampler,
    capture_rng_state,
    restore_rng_state,
)
from training.sft_tokenization import sft_dataset_fingerprint


SFT_RESUME_SCHEMA_VERSION = 1


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
             allow_adamw_fallback: bool = False,
             resume_state: dict | None = None):
    model.to(runtime.device)
    model.train()

    loader_options = dataloader_kwargs(runtime, num_workers)
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

    microbatches_per_epoch = len(train_dataset) // config.sft_batch_size
    if microbatches_per_epoch <= 0:
        raise ValueError("SFT training split is smaller than one full microbatch")
    steps_per_epoch = math.ceil(microbatches_per_epoch / config.sft_grad_accum_steps)
    total_steps = steps_per_epoch * config.sft_epochs
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False
    start_epoch = 0
    epoch_microbatch_offset = 0
    committed_sampler = ResumableBatchSampler(
        len(train_dataset), config.sft_batch_size, drop_last=True
    )
    resume_contract = {
        "dataset_size": len(train_dataset),
        "dataset_fingerprint": sft_dataset_fingerprint(train_dataset),
        "batch_size": config.sft_batch_size,
        "grad_accum_steps": config.sft_grad_accum_steps,
    }

    if resume_state is not None:
        if int(resume_state.get("sft_resume_schema_version", 0)) != SFT_RESUME_SCHEMA_VERSION:
            raise ValueError("SFT checkpoint does not contain supported resume state")
        if resume_state.get("sft_resume_contract") != resume_contract:
            raise ValueError(
                "SFT resume contract differs: "
                f"saved={resume_state.get('sft_resume_contract')}, "
                f"requested={resume_contract}"
            )
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scaler" in resume_state:
            scaler.load_state_dict(resume_state["scaler"])
        sampler_state = resume_state.get("train_sampler")
        if not isinstance(sampler_state, dict):
            raise ValueError("SFT resume checkpoint has no train sampler state")
        committed_sampler = ResumableBatchSampler.from_state_dict(sampler_state)
        if committed_sampler.dataset_size != len(train_dataset):
            raise ValueError("SFT resume dataset size differs from the checkpoint")
        if committed_sampler.batch_size != config.sft_batch_size:
            raise ValueError("SFT resume batch size differs from the checkpoint")
        best_val_loss = float(resume_state.get("best_val_loss", best_val_loss))
        patience_counter = int(resume_state.get("patience_counter", 0))
        global_step = int(resume_state.get("step", 0))
        start_epoch = int(resume_state.get("epoch_index", 0))
        epoch_microbatch_offset = int(resume_state.get("epoch_microbatch_offset", 0))
        if not 0 <= epoch_microbatch_offset <= microbatches_per_epoch:
            raise ValueError("SFT resume microbatch offset is out of range")
        restore_rng_state(resume_state.get("rng_state"))

    current_epoch = start_epoch

    def save_interrupt_checkpoint(path: str) -> None:
        atomic_torch_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
            "optimizer_warning": "fallback_not_recommended"
            if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
            else "",
            "muon_verified": config.muon_verified,
            "sft_resume_schema_version": SFT_RESUME_SCHEMA_VERSION,
            "sft_resume_contract": resume_contract,
            "step": global_step,
            "epoch_index": current_epoch,
            "epoch_microbatch_offset": epoch_microbatch_offset,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "train_sampler": committed_sampler.state_dict(),
            "rng_state": capture_rng_state(),
            "config_schema_version": CHECKPOINT_CONFIG_SCHEMA_VERSION,
            "config": config_to_dict(config),
            "tokenizer": checkpoint_tokenizer_contract(config),
        }, path)
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
        for epoch in range(start_epoch, config.sft_epochs):
            current_epoch = epoch
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pending_grads = False
            pending_microbatches = 0
            pending_losses = []

            actual_sampler = ResumableBatchSampler.from_state_dict(
                committed_sampler.state_dict()
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=actual_sampler,
                **loader_options,
            )
            remaining_microbatches = microbatches_per_epoch - epoch_microbatch_offset
            train_iter = iter(train_loader)
            pbar = tqdm(
                range(remaining_microbatches),
                desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}",
                initial=epoch_microbatch_offset,
                total=microbatches_per_epoch,
            )
            try:
                for _ in pbar:
                    x, y = next(train_iter)
                    x = x.to(runtime.device, non_blocking=non_blocking)
                    y = y.to(runtime.device, non_blocking=non_blocking)
                    with autocast_context(runtime):
                        _, loss = model(x, y)
                        loss = loss / config.sft_grad_accum_steps

                    scaler.scale(loss).backward()
                    pending_grads = True
                    pending_microbatches += 1
                    pending_losses.append(loss.detach())

                    if pending_microbatches == config.sft_grad_accum_steps:
                        committed_microbatches = pending_microbatches
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
                        committed_sampler.advance_batches(committed_microbatches)
                        epoch_microbatch_offset += committed_microbatches
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
                committed_microbatches = pending_microbatches
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
                committed_sampler.advance_batches(committed_microbatches)
                epoch_microbatch_offset += committed_microbatches
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
            print(
                f"Epoch {epoch + 1} | train_loss "
                f"{epoch_loss / max(n_batches, 1):.4f} | val_loss {val_loss:.4f}"
            )
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
            epoch_microbatch_offset = 0
            current_epoch = epoch + 1
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, interrupt_checkpoint_name)
        optimizer.zero_grad(set_to_none=True)
        save_interrupt_checkpoint(interrupt_path)
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
