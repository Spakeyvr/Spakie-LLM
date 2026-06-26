"""SFT fine-tuning loop."""

import math
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime import RuntimeSettings, autocast_context, dataloader_kwargs
from training.dataset import ChatSFTDataset, train_val_split
from training.muon_core import adamw_fallback_warning, should_adamw_fallback
from training.optimizers import configure_torch_optimizer, set_optimizer_lr


def _ensure_contiguous_mps_grads(model: SpakieGPT, runtime: RuntimeSettings) -> None:
    if runtime.device.type != "mps":
        return
    for param in model.parameters():
        if param.grad is not None and not param.grad.is_contiguous():
            param.grad = param.grad.contiguous()


def _scale_partial_sft_grads(model: SpakieGPT, config: SpakieConfig, microbatches_in_step: int) -> None:
    if microbatches_in_step <= 0:
        raise ValueError("microbatches_in_step must be positive")
    if microbatches_in_step == config.sft_grad_accum_steps:
        return
    scale = config.sft_grad_accum_steps / microbatches_in_step
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(scale)


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
) -> object:
    _scale_partial_sft_grads(model, config, microbatches_in_step)
    if config.sft_grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.sft_grad_clip)

    progress = global_step / max(total_steps, 1)
    lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (1 + math.cos(math.pi * progress))
    set_optimizer_lr(optimizer, lr)
    _ensure_contiguous_mps_grads(model, runtime)

    try:
        optimizer.step()
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
            _ensure_contiguous_mps_grads(model, runtime)
            optimizer.step()
        else:
            raise
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

    total_steps = math.ceil(len(train_loader) * config.sft_epochs / config.sft_grad_accum_steps)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    interrupted = False

    try:
        for epoch in range(config.sft_epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pending_grads = False
            pending_microbatches = 0

            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")
            try:
                for batch_idx, (x, y) in enumerate(pbar):
                    x, y = x.to(runtime.device), y.to(runtime.device)
                    with autocast_context(runtime):
                        _, loss = model(x, y)
                        loss = loss / config.sft_grad_accum_steps

                    loss.backward()
                    pending_grads = True
                    pending_microbatches += 1
                    epoch_loss += loss.item() * config.sft_grad_accum_steps
                    n_batches += 1

                    if (batch_idx + 1) % config.sft_grad_accum_steps == 0:
                        optimizer = _sft_optimizer_step(
                            model,
                            optimizer,
                            config,
                            runtime,
                            global_step=global_step,
                            total_steps=total_steps,
                            microbatches_in_step=pending_microbatches,
                            allow_adamw_fallback=allow_adamw_fallback,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        pending_grads = False
                        pending_microbatches = 0
                        global_step += 1

                    pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")
            finally:
                pbar.close()

            if pending_grads:
                optimizer = _sft_optimizer_step(
                    model,
                    optimizer,
                    config,
                    runtime,
                    global_step=global_step,
                    total_steps=total_steps,
                    microbatches_in_step=pending_microbatches,
                    allow_adamw_fallback=allow_adamw_fallback,
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Validation
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(runtime.device), y.to(runtime.device)
                    with autocast_context(runtime):
                        _, loss = model(x, y)
                    val_loss += loss.item()
                    val_count += 1

            val_loss = val_loss / max(val_count, 1)
            print(f"Epoch {epoch + 1} | train_loss {epoch_loss / n_batches:.4f} | val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                ckpt_path = os.path.join(config.checkpoint_dir, best_checkpoint_name)
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
                    "optimizer_warning": "fallback_not_recommended"
                    if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
                    else "",
                    "muon_verified": config.muon_verified,
                    "config": config,
                }, ckpt_path)
                print(f"  -> saved best SFT checkpoint (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.sft_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, interrupt_checkpoint_name)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_kind": getattr(optimizer, "optimizer_kind", config.sft_optimizer),
            "optimizer_warning": "fallback_not_recommended"
            if getattr(optimizer, "optimizer_kind", config.sft_optimizer) == "adamw"
            else "",
            "muon_verified": config.muon_verified,
            "step": global_step,
            "best_val_loss": best_val_loss,
            "config": config,
        }, interrupt_path)
        print(f"\nInterrupted during fine-tuning. Saved checkpoint to {interrupt_path}")

    if interrupted:
        print(f"SFT stopped early. Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"SFT complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss
