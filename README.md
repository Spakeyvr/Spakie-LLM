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

`scripts/prepare_data.py` streams documents from `data/raw/`, including `data/raw/large_corpus/<source>/`, applies quality filters, fastText language ID, and MinHash/LSH near-deduplication, tokenizes in batches, writes token shards under `data/processed/shards/`, and merges them into `data/processed/train.npy` and `data/processed/val.npy`.

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
and automatically starts a background LAN monitor on port `8765`. Open the
printed `http://<your-mac-ip>:8765` URL on your phone while it is on the same
Wi-Fi. The page shows step/token progress, loss, throughput, ETA, checkpoint
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
python3 scripts/monitor_training.py --password "choose-a-long-password"
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

System messages are optional. The default SFT merge omits them, which is usually
better for the smaller presets; pass `--system "..."` only when you intend to
train and infer with that extra control turn.

SFT source data is downloaded to `data/chat_raw/` and merged into `data/chat/train.jsonl`:

```bash
python3 scripts/download_sft_data.py
python3 scripts/prepare_sft.py
python3 scripts/prepare_sft.py --system "Answer clearly and factually."
python3 scripts/prepare_sft.py --assistant-seed-repeats 0
python3 scripts/finetune.py --backend mlx --precision auto
python3 scripts/finetune.py --backend torch --device auto --precision auto
```

The default SFT download uses the enabled sources in `configs/default.yaml`.
The current downloadable defaults are NVIDIA Nemotron instruction-following chat
v3 and Nemotron math v4, each capped at 50,000 examples. To download only
selected sources:

```bash
python3 scripts/download_sft_data.py --sources nemotron_math_v4
```

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

`prepare_sft.py` also adds a small repeated assistant-behavior seed set by
default. This anchors greetings, identity questions, and simple factual answers
so a lightly fine-tuned model is less likely to continue pretraining-style web
text. Set `--assistant-seed-repeats 0` to disable it.

## Optimizer

The default optimizer is Muon, with AdamW fallback only when explicitly allowed:

```bash
python3 scripts/train.py --optimizer muon
python3 scripts/train.py --optimizer adamw --allow-adamw-fallback
python3 scripts/verify_muon.py
```

Muon options include `--muon-adjust-lr-fn {match_rms_adamw,original,none}`, `--muon-ns-steps`, `--muon-momentum`, `--muon-nesterov / --no-muon-nesterov`, and `--muon-qkv-split / --no-muon-qkv-split`.

## Checkpoints and Chat

Checkpoints live under `checkpoints/<preset>/`. Smoke-test outputs live under `smoke_pretrain/` and `smoke_sft/` subdirectories. `pretrain_interrupt.pt` is the rolling checkpoint used by `scripts/train.py --resume`.

Useful commands:

```bash
python3 scripts/chat.py --list-models --device auto --precision auto
python3 scripts/chat.py --model 1 --backend mlx --no-model-prompt
python3 scripts/chat.py --checkpoint checkpoints/300m/best.pt --backend mlx
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
| `92m` | 12 | 768 | 12 | 12 | GELU `d_ff=3072` | 92 | 1 | 92 | 2 | Smallest preset, good for smoke tests and quicker iteration |
| `180m` | 16 | 896 | 14 | 2 | SwiGLU hidden 2048 | 96 | 2 | 32 | 4 | Mid-size MLX-optimized preset |
| `300m` | 10 | 1280 | 20 | 20 | GELU `d_ff=9088` | 128 | 2 | 16 | 2 | Config and pipeline default preset; MLX SFT uses sortish length buckets with padding trim |

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
<|system|>You are Spakie, a helpful assistant.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```

The system turn is optional. `scripts/chat.py` omits it by default unless
`--system` is provided.
