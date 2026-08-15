# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, MLX (`mlx>=0.31.1`) is installed automatically; on CUDA machines, install a matching CUDA wheel of PyTorch separately if the default wheel is wrong.

## Common commands

End-to-end pipeline (data prep → pretrain → SFT) is `scripts/run_pipeline.py`. Individual stages:

```bash
# 1. Download a large pretrain corpus into data/raw/large_corpus/
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only

# 2. Train the tokenizer from cleaned, source-balanced corpus samples
#    Outputs tokenizer/spakie.{model,vocab}
python3 tokenizer/train_tokenizer.py

# 3. Tokenize, filter, near-dedup, and shard into data/processed/{train,val}.npy
python3 scripts/prepare_data.py            # full run
python3 scripts/prepare_data.py --resume   # resume after a partial shard run
python3 scripts/prepare_data.py --dry_run  # estimate token totals without writing arrays

# 4. Pretrain (defaults: --backend mlx, --preset 92m in train.py; pipeline default is 300m)
python3 scripts/train.py --preset 300m --backend mlx --precision auto
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
python3 scripts/train.py --resume                       # latest pretrain_interrupt.pt in preset checkpoint dir
python3 scripts/train.py --smoke                        # short 100-step smoke run

# Optional: preview six fair 100M-token LR/schedule pilots; add --execute to run
python3 scripts/run_pretrain_ablations.py --preset 300m --backend mlx

# 5. Fine-tune on chat JSONL (data/chat/train.jsonl)
python3 scripts/finetune.py --backend mlx --precision auto
python3 scripts/finetune.py --list-models --backend mlx

# 6. Chat with a checkpoint
python3 scripts/chat.py --backend mlx --precision auto
python3 scripts/chat.py --list-models --device auto --precision auto
```

Pipeline runner combining all stages:

```bash
python3 scripts/run_pipeline.py --preset 300m --backend mlx --max-steps 1000
python3 scripts/run_pipeline.py --preset 180m --backend torch --device auto --precision auto
```

## Tests

`unittest` is used directly (no pytest). Run all:

```bash
python3 -m unittest discover -s tests -v
```

Run one module or one test:

```bash
python3 -m unittest tests.test_muon -v
python3 -m unittest tests.test_muon.MuonCoreTests.test_newton_schulz_preserves_shape_and_is_finite
```

Notable tests:
- `tests/test_muon.py` — Muon optimizer math, parameter classification, BF16 tolerance bounds
- `tests/test_mlx_parity.py` — numerical parity between PyTorch and MLX transformer paths
- `tests/test_scaling.py` — preset/config invariants
- `tests/test_runtime.py` — device/precision auto-resolution

A standalone Muon MLX↔PyTorch parity check also runs as part of `train.py` before any full MLX pretraining (skip with `--smoke`):

```bash
python3 scripts/verify_muon.py
```

## Architecture

The codebase has **two parallel backends** (PyTorch and MLX) that share configs, tokenizer, checkpoints, and CLI entry points. Each script dispatches on `--backend {torch,mlx}` to a backend-specific module. When changing model behavior or training logic, almost every change must be mirrored in both halves; the parity test in `tests/test_mlx_parity.py` is the safety net.

Backend mirror map:

| Concern         | Torch                              | MLX                                    |
|-----------------|------------------------------------|----------------------------------------|
| Model           | `model/transformer.py`             | `model/transformer_mlx.py`             |
| Dataset         | `training/dataset.py`              | `training/dataset_mlx.py`              |
| Pretrain loop   | `training/pretrain.py`             | `training/pretrain_mlx.py`             |
| SFT loop        | `training/finetune.py`             | `training/finetune_mlx.py`             |
| Optimizer       | `training/optimizers.py`           | `training/optimizers_mlx.py`           |
| Chat / generate | `inference/chat.py`, `generate.py` | `inference/chat_mlx.py`, `generate_mlx.py` |

Shared infrastructure:
- `configs/default.py` — `SpakieConfig` dataclass + `get_preset_config()` for presets `92m`, `180m`, `300m`. Also defines the corpus source plan, near-dedup/langid filtering knobs, and derived fields (`pretrain_max_steps`, `target_processed_tokens`) refreshed via `refresh_derived_fields()`.
- `runtime/backends.py` — Torch device/precision resolution (`auto` → cuda > mps > cpu; precision `auto` → bf16/cuda, bf16/mps, fp32/cpu) and autocast helpers.
- `runtime/mlx_backend.py` — MLX-specific runtime helpers (compile, prefetch, wired/memory limits).
- `training/muon_core.py` — Backend-agnostic Muon helpers (parameter classification, Newton–Schulz coefficients, BF16 tolerance constants, optimizer/adjust-lr choices, AdamW fallback warning).
- `tokenizer/spakie.model` — SentencePiece tokenizer artifact. Canonical fresh runs use vocab_size 24576 and max_seq_len 2048; retrain it and rebuild processed arrays after changing either contract.

### Data pipeline (`scripts/prepare_data.py`)

Streams documents from `data/raw/` (including `data/raw/large_corpus/<source>/`), applies per-source min-doc-chars, fastText langid (lid.176, auto-downloaded to `data/models/`), MinHash+LSH near-dedup, then SentencePiece-tokenizes into fixed-size shards under `data/processed/shards/` before transactionally merging into `data/processed/{train,val}.npy`. Parallel filtering is consumed in deterministic input-file order so `--resume` can replay the exact prefix. `processed_data_manifest.json` is the commit marker published after both arrays are durable; the pipeline refuses arrays without it. `corpus_source_plan` in `SpakieConfig` is *scaled* at runtime via `scaled_corpus_source_plan()` to hit `target_processed_tokens` (derived from `target_train_tokens` and `train_split_fraction`).

### Optimizer

The default pretraining optimizer is **Muon** (with AdamW fallback gated by `--allow-adamw-fallback`); canonical SFT uses one AdamW epoch. Muon is applied only to "Muon-eligible" parameters (per-tensor classification via `is_muon_parameter_name`); the rest go through AdamW. The MLX Muon implementation must match the Torch one — `verify_muon_for_full_mlx_pretrain()` in `scripts/train.py` runs a parity check before any non-smoke MLX pretrain, and refuses to start if it fails. When changing Newton–Schulz coefficients, NS step count, or parameter classification, update both backends and re-run `tests/test_muon.py` + `scripts/verify_muon.py`.

### Checkpoints

Per-preset directory under `checkpoints/<preset>/`, with `smoke_pretrain/` and `smoke_sft/` subdirs for smoke runs. `checkpoint_search_dirs()` in `configs/default.py` lists fallback dirs used by `chat.py --list-models` and resume logic. New checkpoints store a versioned full `SpakieConfig`; resume restores it before explicit CLI overrides and validates sampler/Muon compatibility. Torch loads with `weights_only=True` unless `--trust-checkpoint` is explicitly supplied for a verified legacy pickle. MLX model tensors are loaded strictly; legacy MLX files without full config metadata require the explicit `--allow-legacy-config` compatibility opt-in. `pretrain_interrupt.pt` is the rolling Torch resume checkpoint written by the pretrain loop; `--resume` picks it up automatically.

### Chat template

```
<|system|>You are Spakie, a helpful assistant.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```

The system turn is optional. Prefer no-system SFT and no-system chat for the
smaller presets unless a run was explicitly trained with system turns.

SFT data must be chat-style JSONL with a top-level `messages` array of `{role, content}` objects. Assistant messages may additionally use `train: false` to remain as context without contributing loss.

## Mac / MPS notes

If torch+MPS hits unsupported ops or memory pressure:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

MLX-specific flags on the train/finetune/chat scripts: `--mlx-compile`, `--mlx-prefetch`, `--mlx-memory-gb`, `--mlx-wired-gb`, `--mlx-profile`.
