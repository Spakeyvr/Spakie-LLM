# Spakie-LLM

A GPT-style language model built from scratch in PyTorch. The repo still trains on local `train.npy` / `val.npy` files, but now includes a scalable corpus pipeline aimed at building a roughly 2B-train-token pretraining run from a single-machine corpus workflow.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch using the official selector for your platform:
- Apple Silicon / Mac: install a recent `torch` build with MPS support
- NVIDIA CUDA: install the matching CUDA wheel for your toolkit
- CPU-only: install the default CPU wheel

If you prefer, you can install PyTorch first and then install the rest:
```bash
pip install torch
pip install -r requirements.txt
```

MLX support in this repo targets Apple Silicon and is validated against `mlx>=0.31.1`.

## Runtime Selection

All runtime entrypoints support backend-aware device and precision flags:
```bash
--device {auto,cuda,mps,cpu}
--precision {auto,fp32,fp16,bf16}
```

Default behavior is Mac-friendly:
- `--device auto` prefers `cuda`, then `mps`, then `cpu`
- `--precision auto` resolves to `bf16` on CUDA, `fp16` on MPS, and `fp32` on CPU

Recommended Apple Silicon usage:
```bash
python3 scripts/train.py --smoke --device auto --precision auto
python3 scripts/finetune.py --smoke --device auto --precision auto
python3 scripts/chat.py --device auto --precision auto
```

The scripts print the resolved device and precision at startup so you can confirm that Mac runs are using `mps` rather than silently falling back to CPU.

For the MLX backend specifically, you can enable rolling timing buckets at existing report boundaries:
```bash
python3 scripts/train.py --backend mlx --mlx-profile
python3 scripts/finetune.py --backend mlx --mlx-profile
```

## Usage

### 1. Add or download training data
Drop `.md` files into `data/raw/`.

You can also scrape open datasets directly:
```bash
python3 scripts/scrape_wiki.py
python3 scripts/scrape_dictionary.py --max 5000
python3 scripts/scrape_open_corpus.py
```

For the larger 2B-token pipeline, download resumable JSONL shards into `data/raw/large_corpus/`:
```bash
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
python3 scripts/download_pretrain_corpus.py --sources fineweb-edu,refinedweb,fineweb,c4,wikipedia_snapshot,stackexchange,open-web-math,arxiv,gutenberg,cosmopedia-v2 --target_tokens_estimate 2105263158 --resume --english_only
```

The downloader writes per-source progress and shard manifests so interrupted runs can continue safely.
`dolma` support remains in the script, but it is not part of the default `all` set because current Hugging Face `datasets` rejects its legacy script loader.
The default source plan now includes supplemental streamable corpora (`fineweb`, `c4`, `open-web-math`, and `cosmopedia-v2`) so a 300m-preset corpus can reach the processed-token target without depending only on slower APIs such as Stack Exchange and arXiv.

### 2. Train tokenizer
```bash
python3 tokenizer/train_tokenizer.py
```

### 3. Prepare data
```bash
python3 scripts/prepare_data.py
```

Useful options:
```bash
python3 scripts/prepare_data.py --dry_run
python3 scripts/prepare_data.py --target_train_tokens 2000000000 --report_path data/processed/corpus_report.json
python3 scripts/prepare_data.py --target_tokens 2105263158 --report_path data/processed/corpus_report.json
python3 scripts/prepare_data.py --source_dirs large_corpus/fineweb-edu,large_corpus/gutenberg --dedup
```

`prepare_data.py` now:
- streams documents from `.md`, `.txt`, and JSONL shards
- performs document-level near-exact dedup
- filters short, noisy, repeated-line, and boilerplate-heavy documents
- tokenizes accepted documents in ordered multicore batches by default
- writes token shards first, then merges them into `data/processed/train.npy` and `data/processed/val.npy`
- emits a corpus report with per-source targets, train/val token totals, and remaining gap to target

By default, `python3 scripts/prepare_data.py` uses a recommended number of
SentencePiece tokenizer threads for the machine. On an 18-core CPU this defaults
to 16 tokenizer threads, leaving a little headroom for I/O and system work. You
can override it with `--tokenizer_threads`.

### 4. Pretrain
```bash
python3 scripts/train.py --preset 180m --device auto --precision auto
python3 scripts/train.py --backend mlx --preset 92m --precision auto --mlx-profile
```

### 5. Fine-tune (optional)
Add chat data to `data/chat/train.jsonl` in the format:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Then run:
```bash
python3 scripts/finetune.py --device auto --precision auto
python3 scripts/finetune.py --backend mlx --precision auto --mlx-profile
```

### 5b. Benchmark MLX training
Use the benchmark harness to compare MLX step throughput without checkpoint writes:
```bash
python3 scripts/benchmark_mlx_training.py --task pretrain --preset 92m --steps 10 --precision auto
python3 scripts/benchmark_mlx_training.py --task pretrain --preset 300m --steps 10 --real-data --precision auto
python3 scripts/benchmark_mlx_training.py --task sft --preset 92m --steps 10 --synthetic --no-prefetch
```

If local training data is missing, the benchmark script falls back to synthetic batches automatically.

### 6. Chat
```bash
python3 scripts/chat.py --device auto --precision auto
python3 scripts/chat.py --model sft_best --device auto --precision auto
python3 scripts/chat.py --list-models --device auto --precision auto
python3 scripts/chat.py --json_mode --device auto --precision auto
python3 scripts/chat.py --temperature 0.5 --top_k 40 --device auto --precision auto
```

## Mac Troubleshooting

If you hit MPS backend limitations or memory pressure on macOS, try these optional environment variables before running a script:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

These are not set automatically by the repo. Use them only when you need to trade strict MPS execution or allocator limits for stability.

## Architecture

| Parameter | Value |
|---|---|
| vocab_size | 8,192 |
| n_layers | 8 |
| n_heads | 8 |
| d_model | 512 |
| d_ff | 2,048 |
| max_seq_len | 512 |
| dropout | 0.1 |

Pre-norm Transformer with learned positional embeddings, weight-tied LM head, GELU activations, and `F.scaled_dot_product_attention` (FlashAttention when available).

## Corpus Notes

- The default train-token target is `2_000_000_000`.
- With the default `0.95` train split, the derived processed-corpus target is `2,105,263,158` tokens.
- The default `180m` pretraining path now derives its step budget from that token goal instead of using a fixed `10,000` steps.
- Baseline from the existing checked-in corpus report before this change: `347,013,932` processed tokens, `535,723,151` estimated tokens from current raw text, and the old default `180m` run consumed `163,840,000` tokens total.
- Storage is expected to stay in the tens to low hundreds of GB range by filtering early and writing resumable shards instead of duplicate full copies.

## Chat Template

```
<|system|>You are Spakie, a helpful assistant.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```
