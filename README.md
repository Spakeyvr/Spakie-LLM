# Spakie-LLM

A GPT-style language model built from scratch in PyTorch. The repo still trains on local `train.npy` / `val.npy` files, but now includes a scalable corpus pipeline aimed at building a roughly 2B-train-token pretraining run from a single-machine corpus workflow.

## Setup

```bash
setup_env.bat
```

Or manually:
```bash
python -m venv venv
venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

### 1. Add or download training data
Drop `.md` files into `data/raw/`.

You can also scrape open datasets directly:
```bash
python scripts/scrape_wiki.py
python scripts/scrape_dictionary.py --max 5000
python scripts/scrape_open_corpus.py
```

For the larger 2B-token pipeline, download resumable JSONL shards into `data/raw/large_corpus/`:
```bash
python scripts/download_pretrain_corpus.py --sources all --resume --english_only
python scripts/download_pretrain_corpus.py --sources fineweb-edu,refinedweb,wikipedia_snapshot,stackexchange,arxiv,gutenberg --target_tokens_estimate 2105263158 --resume --english_only
```

The downloader writes per-source progress and shard manifests so interrupted runs can continue safely.
`dolma` support remains in the script, but it is not part of the default `all` set because current Hugging Face `datasets` rejects its legacy script loader.

Windows wrapper:
```bash
download_pretrain_corpus.bat --sources all --resume --english_only
```

### 2. Train tokenizer
```bash
python tokenizer/train_tokenizer.py
```

### 3. Prepare data
```bash
python scripts/prepare_data.py
```

Useful options:
```bash
python scripts/prepare_data.py --dry_run
python scripts/prepare_data.py --target_train_tokens 2000000000 --report_path data/processed/corpus_report.json
python scripts/prepare_data.py --target_tokens 2105263158 --report_path data/processed/corpus_report.json
python scripts/prepare_data.py --source_dirs large_corpus/fineweb-edu,large_corpus/gutenberg --dedup
```

`prepare_data.py` now:
- streams documents from `.md`, `.txt`, and JSONL shards
- performs document-level near-exact dedup
- filters short, noisy, repeated-line, and boilerplate-heavy documents
- writes token shards first, then merges them into `data/processed/train.npy` and `data/processed/val.npy`
- emits a corpus report with per-source targets, train/val token totals, and remaining gap to target

### 4. Pretrain
```bash
python scripts/train.py --preset 180m
```

### 5. Fine-tune (optional)
Add chat data to `data/chat/train.jsonl` in the format:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Then run:
```bash
python scripts/finetune.py
```

### 6. Chat
```bash
python scripts/chat.py
python scripts/chat.py --model sft_best
python scripts/chat.py --list-models
python scripts/chat.py --json_mode
python scripts/chat.py --temperature 0.5 --top_k 40
```

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
