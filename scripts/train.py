"""CLI entry point for pretraining."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from model.utils import print_model_summary
from training.dataset import PretrainDataset


def main():
    config = SpakieConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = SpakieGPT(config)
    print_model_summary(model)

    # Data
    train_ds = PretrainDataset(os.path.join(config.processed_data_dir, "train.npy"), config.max_seq_len)
    val_ds = PretrainDataset(os.path.join(config.processed_data_dir, "val.npy"), config.max_seq_len)
    print(f"Train sequences: {len(train_ds):,}")
    print(f"Val sequences:   {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=config.pretrain_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.pretrain_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Train
    from training.pretrain import pretrain
    pretrain(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
