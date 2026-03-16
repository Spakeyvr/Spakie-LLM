"""Pretraining loop with cosine LR, gradient accumulation, early stopping."""

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
from model.utils import count_parameters


def get_lr(step: int, config: SpakieConfig) -> float:
    """Cosine schedule with linear warmup, decaying to 10% of peak."""
    min_lr = config.pretrain_lr * 0.1
    if step < config.pretrain_warmup_steps:
        return config.pretrain_lr * step / config.pretrain_warmup_steps
    if step >= config.pretrain_max_steps:
        return min_lr
    progress = (step - config.pretrain_warmup_steps) / (config.pretrain_max_steps - config.pretrain_warmup_steps)
    return min_lr + 0.5 * (config.pretrain_lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model: SpakieGPT, config: SpakieConfig) -> torch.optim.AdamW:
    """AdamW with weight decay only on 2D params (not biases, norms, embeddings)."""
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay_params, "weight_decay": config.pretrain_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.pretrain_lr, betas=(0.9, 0.95), fused=True)


@torch.no_grad()
def evaluate(model: SpakieGPT, val_loader: DataLoader, config: SpakieConfig, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= config.pretrain_eval_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def pretrain(model: SpakieGPT, train_loader: DataLoader, val_loader: DataLoader,
             config: SpakieConfig, device: torch.device):
    model.to(device)
    model.train()
    optimizer = configure_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bfloat16 doesn't need scaling

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    tokens_processed = 0
    start_time = time.time()
    interrupted = False

    train_iter = iter(train_loader)
    pbar = tqdm(total=config.pretrain_max_steps, desc="Pretrain")

    try:
        while global_step < config.pretrain_max_steps:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(config.pretrain_grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / config.pretrain_grad_accum_steps

                loss.backward()
                accum_loss += loss.item()
                tokens_processed += x.numel()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.pretrain_grad_clip)

            # Update LR
            lr = get_lr(global_step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            global_step += 1
            pbar.update(1)

            # Eval
            if global_step % config.pretrain_eval_interval == 0:
                val_loss = evaluate(model, val_loader, config, device)
                elapsed = time.time() - start_time
                tok_per_sec = tokens_processed / elapsed
                pbar.write(
                    f"step {global_step} | train_loss {accum_loss:.4f} | val_loss {val_loss:.4f} | "
                    f"lr {lr:.2e} | tok/s {tok_per_sec:.0f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    ckpt_path = os.path.join(config.checkpoint_dir, "pretrain_best.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "val_loss": val_loss,
                        "config": config,
                    }, ckpt_path)
                    pbar.write(f"  -> saved best checkpoint (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= config.pretrain_patience:
                        pbar.write(f"Early stopping at step {global_step}")
                        break
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = os.path.join(config.checkpoint_dir, "pretrain_interrupt.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
            "best_val_loss": best_val_loss,
            "config": config,
        }, interrupt_path)
        pbar.write(f"\nInterrupted at step {global_step}. Saved checkpoint to {interrupt_path}")
    finally:
        pbar.close()

    if interrupted:
        print(f"Pretraining stopped early. Best val loss so far: {best_val_loss:.4f}")
    else:
        print(f"Pretraining complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss
