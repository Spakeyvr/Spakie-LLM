# Spakie-LLM

Spakie-LLM is a GPT-style language model project with both PyTorch and MLX runtime paths. It includes tokenizer training, corpus scraping and preprocessing, pretraining, optional SFT fine-tuning, and chat inference from the same codebase.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is the main dependency for the torch backend. On Apple Silicon, install an MPS-enabled build; on CUDA machines, install the matching CUDA wheel for your toolkit. MLX is enabled on Apple Silicon via `mlx>=0.31.1`.

## Project Layout

- `scripts/` - training, preprocessing, scraping, benchmark, and pipeline entry points
- `training/` - dataset, optimizer, pretraining, and fine-tuning logic
- `model/` - PyTorch and MLX Transformer implementations
- `runtime/` - device, precision, and MLX runtime helpers
- `tokenizer/` - SentencePiece tokenizer training and wrapper code
- `inference/` - chat loops for both backends
- `configs/default.py` - preset definitions and corpus planning defaults

## Quick Start

1. Train a tokenizer
```bash
python3 tokenizer/train_tokenizer.py
```

2. Prepare data
```bash
python3 scripts/prepare_data.py
```

3. Train a model
```bash
python3 scripts/train.py --preset 300m --backend mlx --precision auto
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
```

4. Fine-tune on chat data
```bash
python3 scripts/finetune.py --backend mlx --precision auto
python3 scripts/finetune.py --backend torch --device auto --precision auto
```

5. Chat with a checkpoint
```bash
python3 scripts/chat.py --backend mlx --precision auto
python3 scripts/chat.py --backend torch --device auto --precision auto
```

## Runtime Selection

The torch entry points support:

```bash
--device {auto,cuda,mps,cpu}
--precision {auto,fp32,fp16,bf16}
```

Default behavior is Mac-friendly:

- `--device auto` prefers `cuda`, then `mps`, then `cpu`
- `--precision auto` resolves to `bf16` on CUDA, `fp16` on MPS, and `fp32` on CPU

For Apple Silicon, the repo also supports MLX-specific flags:

```bash
--mlx-compile
--mlx-prefetch
--mlx-memory-gb <value>
--mlx-wired-gb <value>
--mlx-profile
```

## Data Sources

You can add local `.md`, `.txt`, and `.jsonl` files under `data/raw/`, or use the built-in download and scrape scripts:

```bash
python3 scripts/scrape_wiki.py
python3 scripts/scrape_dictionary.py --max 5000
python3 scripts/scrape_open_corpus.py
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
```

`scripts/prepare_data.py` streams documents, filters low-quality text, deduplicates near-identical documents, tokenizes in batches, and writes `data/processed/train.npy` and `data/processed/val.npy`.

If preparation is interrupted after token shards are written, resume with:

```bash
python3 scripts/prepare_data.py --resume
```

## Pretraining and SFT

Pretraining:

```bash
python3 scripts/train.py --preset 92m --device auto --precision auto
python3 scripts/train.py --preset 300m --backend mlx --precision auto --mlx-profile
```

Pipeline runner:

```bash
python3 scripts/run_pipeline.py --preset 300m --backend mlx --max-steps 1000
python3 scripts/run_pipeline.py --preset 180m --backend torch --device auto --precision auto
```

Fine-tuning expects chat-style JSONL records like:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Chat data defaults to `data/chat/train.jsonl`.

## Checkpoints

The chat and fine-tuning commands look for checkpoints under the preset checkpoint directory, with smoke-test outputs in `smoke_pretrain/` and `smoke_sft/`.

Useful commands:

```bash
python3 scripts/chat.py --list-models --device auto --precision auto
python3 scripts/finetune.py --list-models --backend mlx
python3 scripts/train.py --resume
```

## Model Presets

The repo currently supports these presets:

| Preset | Layers | `d_model` | Heads | `d_ff` | Pretrain batch | Grad accum | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `92m` | 12 | 768 | 12 | 3072 | 64 | 1 | Smallest preset, good for smoke tests and quicker iteration |
| `180m` | 16 | 896 | 14 | 3584 | 96 | 2 | Uses activation checkpointing |
| `300m` | 24 | 1024 | 16 | 4096 | 64 | 2 | Default preset, uses activation checkpointing |

Shared model defaults:

- `vocab_size = 16384`
- `max_seq_len = 512`
- `dropout = 0.1`
- learned positional embeddings
- weight-tied LM head
- GELU activations
- scaled dot-product attention

## Mac Troubleshooting

If you hit MPS backend limitations or memory pressure on macOS, these environment variables can help:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Chat Template

```
<|system|>You are Spakie, a helpful assistant.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```
