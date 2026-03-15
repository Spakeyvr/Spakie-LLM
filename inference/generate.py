"""Autoregressive generation with top-k/top-p sampling and JSON mode."""

import json

import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer


def sample_top_k_top_p(logits: torch.Tensor, temperature: float = 0.8,
                        top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and top-p filtering."""
    logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        vals, _ = torch.topk(logits, top_k)
        logits[logits < vals[..., -1:]] = float("-inf")

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model: SpakieGPT, tokenizer: SpakieTokenizer, prompt_ids: list[int],
             max_new_tokens: int = 256, temperature: float = 0.8,
             top_k: int = 50, top_p: float = 0.9,
             device: torch.device | str = "cuda") -> list[int]:
    """Autoregressive generation. Returns generated token IDs (excluding prompt)."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        # Crop to max_seq_len
        idx_cond = idx[:, -model.config.max_seq_len:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx_cond)

        next_logits = logits[:, -1, :]
        next_id = sample_top_k_top_p(next_logits, temperature, top_k, top_p)

        token = next_id.item()
        if token == tokenizer.eos_id:
            break

        generated.append(token)
        idx = torch.cat([idx, next_id], dim=1)

    return generated


def generate_json(model: SpakieGPT, tokenizer: SpakieTokenizer, prompt_ids: list[int],
                  max_new_tokens: int = 512, temperature: float = 0.5,
                  top_k: int = 50, top_p: float = 0.9,
                  device: torch.device | str = "cuda", max_retries: int = 3) -> str | None:
    """Generate and validate JSON output. Retries up to max_retries times."""
    # Prepend <|json|> token
    json_prompt = [tokenizer.json_id] + prompt_ids

    for attempt in range(max_retries):
        ids = generate(model, tokenizer, json_prompt, max_new_tokens, temperature, top_k, top_p, device)
        text = tokenizer.decode(ids).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                temperature *= 0.8  # Lower temperature on retry
            continue

    return None
