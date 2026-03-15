"""Preprocess .md files into tokenized .npy files for pretraining."""

import glob
import os
import re

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from tokenizer.train_tokenizer import SpakieTokenizer


def clean_text(text: str) -> str:
    """Light cleaning: strip HTML tags, normalize whitespace."""
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML
    text = re.sub(r"\r\n", "\n", text)             # normalize line endings
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excessive newlines
    return text.strip()


def prepare_data(config: SpakieConfig | None = None):
    config = config or SpakieConfig()
    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")

    md_files = sorted(glob.glob(os.path.join(config.raw_data_dir, "**", "*.md"), recursive=True))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {config.raw_data_dir}")

    print(f"Found {len(md_files)} markdown files")

    all_ids = []
    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            text = clean_text(f.read())
        if text:
            ids = tokenizer.encode(text)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_id)  # EOS between docs

    all_ids = np.array(all_ids, dtype=np.uint16)
    print(f"Total tokens: {len(all_ids):,}")

    # 95/5 split
    split_idx = int(len(all_ids) * 0.95)
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    os.makedirs(config.processed_data_dir, exist_ok=True)
    train_path = os.path.join(config.processed_data_dir, "train.npy")
    val_path = os.path.join(config.processed_data_dir, "val.npy")

    np.save(train_path, train_ids)
    np.save(val_path, val_ids)

    print(f"Train: {len(train_ids):,} tokens -> {train_path}")
    print(f"Val:   {len(val_ids):,} tokens -> {val_path}")


if __name__ == "__main__":
    prepare_data()
