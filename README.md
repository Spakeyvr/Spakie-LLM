# Spakie-LLM

Spakie-LLM is a GPT-style language model project with parallel PyTorch and MLX runtime paths. It includes tokenizer training, corpus download/scraping and preprocessing, pretraining, SFT fine-tuning, checkpointed chat inference, and basic evaluation tooling from the same codebase.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is the main dependency for the torch backend. On Apple Silicon, MLX is installed via the conditional `mlx>=0.31.1` requirement; on CUDA machines, install a PyTorch wheel that matches your CUDA toolkit if the default wheel is not appropriate.

## Project Layout

- `scripts/` - training, preprocessing, scraping, data download, evaluation, benchmark, and pipeline entry points
- `training/` - dataset, optimizer, pretraining, and fine-tuning logic
- `model/` - PyTorch and MLX Transformer implementations
- `runtime/` - device, precision, and MLX runtime helpers
- `tokenizer/` - SentencePiece tokenizer training and wrapper code
- `inference/` - chat and generation loops for both backends
- `configs/default.py` and `configs/default.yaml` - preset definitions, paths, optimizer defaults, and corpus planning defaults
- `tests/` - `unittest` coverage for runtime resolution, config scaling, Muon, and MLX/PyTorch parity

## Quick Start

1. Train a tokenizer from files under `data/raw/`:

```bash
python3 tokenizer/train_tokenizer.py
```

Tokenizer samples are streamed in deterministic round-robin order across
discovered corpus sources, so the five-million-example cap cannot exclude
later source directories merely because of lexical path order.

2. Download or add pretraining data:

```bash
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
```

3. Prepare pretraining arrays:

```bash
python3 scripts/prepare_data.py
```

4. Train a model:

```bash
python3 scripts/train.py --preset 300m --backend mlx --precision auto
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
```

5. Build SFT data and fine-tune:

```bash
python3 scripts/download_sft_data.py
python3 scripts/prepare_sft.py
python3 scripts/finetune.py --backend mlx --precision auto
```

6. Chat with a checkpoint:

```bash
python3 scripts/chat.py --backend mlx --precision auto
python3 scripts/chat.py --backend torch --device auto --precision auto
```

`scripts/train.py` defaults to `--backend mlx --preset 92m`. The shared config default preset and pipeline default are `300m`.

## Runtime Selection

Torch entry points support:

```bash
--device {auto,cuda,mps,cpu}
--precision {auto,fp32,fp16,bf16}
```

Default runtime behavior:

- `--device auto` prefers `cuda`, then `mps`, then `cpu`
- `--precision auto` resolves to `bf16` on CUDA, `bf16` on MPS, and `fp32` on CPU

Common MLX runtime flags:

```bash
--mlx-compile / --no-mlx-compile
--mlx-prefetch / --no-mlx-prefetch
--mlx-memory-gb <value>
--mlx-wired-gb <value>
--mlx-cache-gb <value>
--mlx-profile
```

MLX pretraining also supports `--mlx-vmap-accum-step /
--no-mlx-vmap-accum-step`.

## Data Sources

You can add local `.md`, `.txt`, and `.jsonl` files under `data/raw/`, or use the built-in download and scrape scripts:

```bash
python3 scripts/scrape_wiki.py
python3 scripts/scrape_dictionary.py --max 5000
python3 scripts/scrape_open_corpus.py
python3 scripts/download_pretrain_corpus.py --sources all --resume --english_only
```

The downloader streams Gutenberg, Stack Exchange, and arXiv from bulk corpus
snapshots rather than rate-limited public APIs. Python-Edu metadata is resolved
through persistent concurrent HTTPS connections to Software Heritage's public
S3 bucket; rows that are too short or already accepted are rejected before the
content request. By default, each Hugging Face source keeps four input shards
active (`--hf-workers-per-source`) and Python-Edu uses up to thirty-two persistent
content fetchers (`--item-workers`, scaled down on smaller machines). New progress
files store exact Hugging Face stream state, so `--resume` continues at the saved
input shard instead of replaying every earlier row. Near-complete progress files
from the old row-counter format skip their multi-million-row replay and fill the
small remaining tail from a source with a direct cursor. Progress is labelled
`Accepted corpus`; the displayed rate is accepted estimated tokens over the
last 15 seconds, so retries, filtering, and other zero-progress time reduce it
instead of leaving an earlier burst rate on screen. If a requested source is
exhausted or unavailable, its shortfall is filled from another requested
streaming source; use `--no-redistribute-shortfall` to preserve strict
per-source quotas instead. Ctrl+C gives active workers five seconds to flush
their checkpoints, then exits without waiting for blocked HTTP retries; pressing
Ctrl+C again skips the grace period. Sources already at their saved target are
skipped before their potentially large resume indexes are loaded.

`scripts/prepare_data.py` streams documents from `data/raw/`, including `data/raw/large_corpus/<source>/`, applies quality filters, fastText language ID, and MinHash/LSH near-deduplication, tokenizes in deterministic input order, writes token shards under `data/processed/shards/`, and transactionally merges them into `data/processed/train.npy` and `data/processed/val.npy`. A `processed_data_manifest.json` commit marker is published only after both arrays are complete and durable; the pipeline will not train from arrays without that marker.

Useful prepare commands:

```bash
python3 scripts/prepare_data.py --resume
python3 scripts/prepare_data.py --dry_run
python3 scripts/prepare_data.py --target_train_tokens 100000000
python3 scripts/prepare_data.py --source_dirs large_corpus,wiki
python3 scripts/prepare_data.py --workers 1
```

## Pretraining and SFT

Pretraining:

```bash
python3 scripts/train.py --preset 92m --backend mlx --precision auto --smoke
python3 scripts/train.py --preset 300m --backend mlx --precision auto --mlx-profile
python3 scripts/train.py --preset 300m --backend torch --device auto --precision auto
```

Useful training options:

```bash
--max-steps <steps>
--target_tokens <tokens>
--resume
--resume-from <checkpoint>
--additional-steps <steps>
--output-dir <dir>
--eval-interval <steps>
--eval-batches <batches>
--checkpoint-interval <steps>
```

Training writes a live status file to `checkpoints/<preset>/training_status.json`
and automatically starts a background monitor on port `8765`. Without a
password it binds only to `127.0.0.1`. To open it from another device on your
LAN, set `MONITOR_PASSWORD` before training; the authenticated monitor then
binds to the LAN address printed at startup. The page shows step/token progress, loss, throughput, ETA, checkpoint
paths, MLX memory when available, and a prompt box for querying the best
available checkpoint. Pretrain checkpoints use raw continuation mode; SFT
checkpoints use the same chat template path as `scripts/chat.py`. A monitor
process started by training is stopped automatically when that training process
exits.

For a public IPv6 address, wrap the address in square brackets:

```text
http://[2001:db8::1234]:8765
```

Public access still requires the network path to allow inbound port `8765`
(router/firewall/ISP), and password protection does not encrypt plain HTTP.

You can also start the monitor manually:

```bash
python3 scripts/monitor_training.py
```

Use `--status-file <path>` to pin the monitor to a specific run, or
`--checkpoint-dir <dir>` to scan a different checkpoint tree. Set
`SPAKIE_MONITOR=0` to disable training autostart, or `SPAKIE_MONITOR_PORT=9000`
to use a different port. Set `SPAKIE_MONITOR_PROMPT_TIMEOUT=300` if prompt
generation needs more than the default 180 seconds.

For password protection, set the password before starting training:

```bash
MONITOR_PASSWORD="choose-a-long-password" python3 scripts/train.py --backend mlx
```

Manual monitor starts also accept a flag:

```bash
python3 scripts/monitor_training.py --host :: --password "choose-a-long-password"
```

When password protection is enabled, the monitor serves only the login page or a
401 response until the password is accepted. If you expose this through a public
IP, put HTTPS/VPN in front of it; plain HTTP still sends the password over the
network unencrypted.

Pipeline runner:

```bash
python3 scripts/run_pipeline.py --preset 300m --backend mlx --max-steps 1000
python3 scripts/run_pipeline.py --preset 180m --backend torch --device auto --precision auto
python3 scripts/run_pipeline.py --preset 300m --backend mlx --skip-sft
```

Fine-tuning expects chat-style JSONL records like:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Assistant messages may set `"train": false` to remain in the conversation as
context without contributing labels. If omitted, assistant turns are supervised
as before. The Nemotron Chat v3 importer uses this for all but the final target.

System messages are optional. The default SFT merge omits them, which is usually
better for the smaller presets; pass `--system "..."` only when you intend to
train and infer with that extra control turn.

SFT source data is downloaded to `data/chat_raw/` and merged into `data/chat/train.jsonl`:

```bash
python3 scripts/download_sft_data.py
python3 scripts/build_sft_seed_data.py
python3 scripts/prepare_sft.py
python3 scripts/prepare_sft.py --system "Answer clearly and factually."
python3 scripts/finetune.py --backend mlx --precision auto
python3 scripts/finetune.py --backend torch --device auto --precision auto
```

The default SFT download uses the enabled sources in `configs/default.yaml`.
The current downloadable defaults target the 180M model: up to 40,000
quality-stratified SmolTalk rows from a 150,000-row raw sample, 8,000 No Robots,
6,000 SciQ, 5,000 SQuAD, 3,000 BoolQ, 5,000 Nemotron instruction-following chat,
and 3,000 TriviaQA rows, plus uncapped local `DeepSeek-distill-V2` and `custom`
files. Preparation removes non-English, conflicting-identity, refusal, review-
annotation, and over-context examples. Advanced Nemotron math is intentionally
not part of this SFT pipeline. Limits are
applied to usable converted examples rather than raw rows. To download only selected sources:

`build_sft_seed_data.py` writes the small permanent local sources separately so
they remain easy to inspect and version: `spakie_180m_identity.jsonl`,
`assistant_behavior.jsonl`, `anti_echo.jsonl`, and `factual_repairs.jsonl`.
They live in `data/chat_raw/` and are merged exactly like downloaded sources.

For a small targeted SFT/eval set instead of downloaded SFT sources:

```bash
python3 scripts/build_targeted_data.py
```

This writes directly to `data/chat/train.jsonl` and `data/eval/`.

Useful SFT options:

```bash
--train-jsonl <path>
--source-checkpoint <checkpoint>
--output-name <filename>
--max-examples <count>
--epochs <count>
--lr <value>
--list-models
--no-model-prompt
```

On MLX, SFT batches are length-bucketed with right-padding bucket trim by
default. This keeps the transformer math dense and unchanged while avoiding a
large amount of padded-token work on chat datasets. The conservative defaults
are `--sft-sampler sortish`, `--sft-bucket-multiple 128`, and no SFT packing or
varlen attention.

The permanent assistant-behavior and identity sources anchor greetings,
Spakie-180M identity questions, direct answers, and simple factual responses so
a lightly fine-tuned model is less likely to continue pretraining-style web
text or invent a human occupation.

## Optimizer

The default optimizer is Muon, with AdamW fallback only when explicitly allowed:

```bash
python3 scripts/train.py --optimizer muon
python3 scripts/train.py --optimizer adamw --allow-adamw-fallback
python3 scripts/verify_muon.py
```

Muon options include `--muon-adjust-lr-fn {match_rms_adamw,original,none}`, `--muon-ns-steps`, `--muon-momentum`, `--muon-nesterov / --no-muon-nesterov`, and `--muon-qkv-split / --no-muon-qkv-split`.

## Checkpoints and Chat

Checkpoints live under `checkpoints/<preset>/`. Smoke-test outputs live under `smoke_pretrain/` and `smoke_sft/` subdirectories. New Torch and MLX checkpoints store a complete versioned configuration, and every model load validates all tensor keys and shapes. Torch uses PyTorch's restricted loader by default; `--trust-checkpoint` is required for a legacy pickle that contains custom Python objects. MLX files created before full config metadata are refused by default; `--allow-legacy-config` is an explicit inexact compatibility opt-in. `pretrain_interrupt.*` is the rolling resume checkpoint; every successful run also atomically publishes `pretrain_final.*`, even when it ends before an evaluation boundary.

Useful commands:

```bash
python3 scripts/chat.py --list-models --device auto --precision auto
python3 scripts/chat.py --model 1 --backend mlx --no-model-prompt
python3 scripts/chat.py --checkpoint checkpoints/300m/pretrain_final.safetensors --backend mlx
python3 scripts/chat.py --json_mode --system "Answer as JSON."
python3 scripts/finetune.py --list-models --backend mlx
python3 scripts/train.py --resume
```

The default chat tokenizer path is `tokenizer/spakie.model`.

## Evaluation and Tests

Run the basic QA evaluator:

```bash
python3 scripts/build_targeted_data.py
python3 scripts/eval_basic_qa.py --backend mlx --preset 300m
python3 scripts/eval_basic_qa.py --backend torch --device auto --precision auto
```

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Notable tests:

- `tests/test_muon.py` - Muon optimizer math, parameter classification, and BF16 tolerance bounds
- `tests/test_mlx_parity.py` - numerical parity between PyTorch and MLX transformer paths
- `tests/test_scaling.py` - preset/config invariants and data preparation behavior
- `tests/test_runtime.py` - device/precision auto-resolution

## Model Presets

The repo currently supports these presets:

| Preset | Layers | `d_model` | Q heads | KV heads | MLP | Pretrain batch | Grad accum | SFT batch | SFT grad accum | Notes |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| `92m` | 12 | 768 | 12 | 4 | GELU `d_ff=3072` | 92 | 1 | 92 | 2 | Smallest preset, good for smoke tests and quicker iteration |
| `180m` | 16 | 896 | 14 | 2 | SwiGLU hidden 2048 | 96 | 2 | 64 | 4 | Mid-size MLX-optimized preset |
| `180m_gqa4` | 16 | 896 | 16 | 4 | SwiGLU hidden 2048 | 96 | 2 | 64 | 4 | Short-run 4-KV-head architecture ablation |
| `180m_deep` | 24 | 768 | 12 | 4 | SwiGLU hidden 1536 | 72 | 2 | 48 | 4 | Short-run deep/thin architecture ablation |
| `300m` | 10 | 1280 | 20 | 4 | GELU `d_ff=9088` | 64 | 3 | 16 | 2 | Config and pipeline default preset; MLX SFT uses sortish length buckets with padding trim |

Shared model defaults:

- `vocab_size = 16384`
- `max_seq_len = 512`
- `dropout = 0.0`
- `bias = false`
- learned positional embeddings
- weight-tied LM head
- GELU or SwiGLU MLPs, depending on preset
- scaled dot-product attention, with grouped-query attention where configured
- activation checkpointing disabled by default for all current presets

## Balanced Pretraining Corpus

The default corpus target is 10B training tokens (about 10.53B processed with
the 95/5 split), with `max_seq_len=512`. The source plan is intentionally
balanced by capability domain:

| Domain | Target share |
|---|---:|
| FineWeb-Edu | 32% |
| General filtered web | 16% |
| Wikipedia/reference | 17% |
| Math (`FineMath-4+` + OpenWebMath) | 12% |
| Educational Python code | 10% |
| Books | 6% |
| arXiv + StackExchange | 2% |
| Cosmopedia synthetic education | 5% |

Download and prepare a fresh generation; existing processed arrays retain the
old mixture until rebuilt:

```bash
python3 scripts/download_pretrain_corpus.py --sources all --resume --english-only
python3 scripts/prepare_data.py
```

Before a full run, compare the architecture variants at the same token budget:

```bash
python3 scripts/train.py --preset 180m --backend mlx --target-tokens 300000000
python3 scripts/train.py --preset 180m_gqa4 --backend mlx --target-tokens 300000000
python3 scripts/train.py --preset 180m_deep --backend mlx --target-tokens 300000000
```

Use `scripts/eval_general_capability.py` after each matched run. Its fixed suite
reports results by category (math, factual recall, instruction following,
formatting, and edge cases), so architecture selection is not based on aggregate
validation loss alone.

## Architecture Notes

The codebase has two parallel backends that share configs, tokenizer, checkpoints, and CLI entry points.

| Concern | Torch | MLX |
|---|---|---|
| Model | `model/transformer.py` | `model/transformer_mlx.py` |
| Dataset | `training/dataset.py` | `training/dataset_mlx.py` |
| Pretrain loop | `training/pretrain.py` | `training/pretrain_mlx.py` |
| SFT loop | `training/finetune.py` | `training/finetune_mlx.py` |
| Optimizer | `training/optimizers.py` | `training/optimizers_mlx.py` |
| Chat / generate | `inference/chat.py`, `inference/generate.py` | `inference/chat_mlx.py`, `inference/generate_mlx.py` |

When changing model behavior or training logic, keep the Torch and MLX paths aligned and re-run the parity tests.

## Mac Troubleshooting

If torch+MPS hits unsupported ops or memory pressure on macOS, these environment variables can help:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## Chat Template

```text
<|system|>You are Spakie-180M, a helpful AI language model.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```

The system turn is optional. `scripts/chat.py` omits it by default unless
`--system` is provided.
