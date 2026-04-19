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
from runtime import RuntimeSettings, autocast_context, dataloader_kwargs, optimizer_kwargs
from training.dataset import ChatSFTDataset, train_val_split


def finetune(model: SpakieGPT, train_dataset: ChatSFTDataset, val_dataset,
             config: SpakieConfig, runtime: RuntimeSettings,
             num_workers: int = 2,
             best_checkpoint_name: str = "sft_best.pt",
             interrupt_checkpoint_name: str = "sft_interrupt.pt"):
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

    # Optimizer: weight decay on 2D params only
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.sft_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=config.sft_lr, betas=(0.9, 0.95), **optimizer_kwargs(runtime))

    total_steps = len(train_loader) * config.sft_epochs // config.sft_grad_accum_steps
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

            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")
            try:
                for batch_idx, (x, y) in enumerate(pbar):
                    x, y = x.to(runtime.device), y.to(runtime.device)
                    with autocast_context(runtime):
                        _, loss = model(x, y)
                        loss = loss / config.sft_grad_accum_steps

                    loss.backward()
                    epoch_loss += loss.item() * config.sft_grad_accum_steps
                    n_batches += 1

                    if (batch_idx + 1) % config.sft_grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.sft_grad_clip)

                        # Cosine LR
                        progress = global_step / max(total_steps, 1)
                        lr = config.sft_lr * 0.1 + 0.5 * config.sft_lr * 0.9 * (1 + math.cos(math.pi * progress))
                        for pg in optimizer.param_groups:
                            pg["lr"] = lr

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                    pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")
            finally:
                pbar.close()

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
