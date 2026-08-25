"""Autoregressive generation with top-k/top-p sampling and JSON mode."""

import json

import torch
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.transformer import SpakieGPT
from runtime import RuntimeSettings, autocast_context
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.generation_utils import (
    apply_repetition_penalty,
    mask_out_of_tokenizer_vocab,
    prepare_generation_prompt,
    validate_sampling_parameters,
)


def sample_top_k_top_p(logits: torch.Tensor, temperature: float = 0.8,
                        top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and top-p filtering."""
    validate_sampling_parameters(temperature, top_k, top_p)
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
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
             repetition_penalty: float = 1.2,
             runtime: RuntimeSettings | None = None,
             device: torch.device | str | None = None,
             stop_on_special_tokens: bool = True,
             ban_special_tokens: bool = True) -> list[int]:
    """Autoregressive generation. Returns generated token IDs (excluding prompt)."""
    if max_new_tokens <= 0:
        return []
    validate_sampling_parameters(temperature, top_k, top_p)
    model.eval()
    if runtime is None:
        runtime_device = torch.device(device) if device is not None else next(model.parameters()).device
        runtime = RuntimeSettings(device=runtime_device, precision="fp32")
    prompt_ids = prepare_generation_prompt(
        prompt_ids,
        max_seq_len=model.config.max_seq_len,
        max_new_tokens=max_new_tokens,
    )
    context_tokens = list(prompt_ids)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
    generated = []
    generated_seen: set[int] = set()

    # Stop tokens: EOS + all chat role tokens
    stop_tokens = {
        tokenizer.eos_id,
        tokenizer.user_id,
        tokenizer.assistant_id,
        tokenizer.system_id,
    }

    with autocast_context(runtime):
        logits, _, cache = model(idx, cache_offset=0, return_cache=True)
    cache_offset = idx.shape[1]

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :].clone()
        mask_out_of_tokenizer_vocab(
            next_logits,
            int(getattr(tokenizer, "vocab_size", next_logits.shape[-1])),
        )

        # Repetition penalty: reduce probability of tokens generated in this
        # answer only. Penalizing prompt tokens makes factual answers avoid
        # repeating entities from the user's question, e.g. country names.
        apply_repetition_penalty(next_logits[0], generated_seen, repetition_penalty)

        if ban_special_tokens:
            next_logits[0, tokenizer.user_id] = float("-inf")
            next_logits[0, tokenizer.assistant_id] = float("-inf")
            next_logits[0, tokenizer.system_id] = float("-inf")
            next_logits[0, tokenizer.json_id] = float("-inf")
            next_logits[0, tokenizer.pad_id] = float("-inf")

        next_id = sample_top_k_top_p(next_logits, temperature, top_k, top_p)

        token = next_id.item()
        if stop_on_special_tokens and token in stop_tokens:
            break

        generated.append(token)
        generated_seen.add(token)
        context_tokens.append(token)
        if len(generated) >= max_new_tokens:
            break

        if cache_offset + 1 <= model.config.max_seq_len:
            idx = next_id
            with autocast_context(runtime):
                logits, _, cache = model(
                    idx,
                    cache=cache,
                    cache_offset=cache_offset,
                    return_cache=True,
                )
            cache_offset += 1
        else:
            window = context_tokens[-model.config.max_seq_len :]
            idx = torch.tensor([window], dtype=torch.long, device=runtime.device)
            with autocast_context(runtime):
                logits, _, cache = model(idx, cache_offset=0, return_cache=True)
            cache_offset = len(window)

    return generated


def generate_json(model: SpakieGPT, tokenizer: SpakieTokenizer, prompt_ids: list[int],
                  max_new_tokens: int = 512, temperature: float = 0.5,
                  top_k: int = 50, top_p: float = 0.9,
                  runtime: RuntimeSettings | None = None,
                  device: torch.device | str | None = None,
                  max_retries: int = 3) -> str | None:
    """Generate and validate JSON output. Retries up to max_retries times."""
    # Prepend <|json|> token
    json_prompt = [tokenizer.json_id] + prompt_ids

    for attempt in range(max_retries):
        ids = generate(
            model,
            tokenizer,
            json_prompt,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            runtime=runtime,
            device=device,
        )
        text = tokenizer.decode(ids).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                temperature *= 0.8  # Lower temperature on retry
            continue

    return None
