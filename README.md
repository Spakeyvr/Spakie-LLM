# Spakie-LLM

A ~30M parameter GPT-style language model built from scratch in PyTorch. Designed to train on small `.md` datasets on a single GPU (e.g., RTX 4070 Ti).

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

### 1. Add training data
Drop `.md` files into `data/raw/`.

You can also scrape open datasets directly:
```bash
python scripts/scrape_wiki.py
python scripts/scrape_dictionary.py --max 5000
python scripts/scrape_open_corpus.py
```

### 2. Train tokenizer
```bash
python tokenizer/train_tokenizer.py
```

### 3. Prepare data
```bash
python scripts/prepare_data.py
```

### 4. Pretrain
```bash
python scripts/train.py
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

## Chat Template

```
<|system|>You are Spakie, a helpful assistant.<eos>
<|user|>What is Python?<eos>
<|assistant|>Python is a programming language.<eos>
```
