"""CLI chat REPL."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.generate import generate, generate_json


DEFAULT_SYSTEM = "You are Spakie, a helpful assistant."


def build_prompt_ids(tokenizer: SpakieTokenizer, history: list[dict], system_msg: str) -> list[int]:
    """Build token IDs from conversation history using chat template."""
    ids = [tokenizer.system_id] + tokenizer.encode(system_msg) + [tokenizer.eos_id]

    for msg in history:
        if msg["role"] == "user":
            ids += [tokenizer.user_id] + tokenizer.encode(msg["content"]) + [tokenizer.eos_id]
        elif msg["role"] == "assistant":
            ids += [tokenizer.assistant_id] + tokenizer.encode(msg["content"]) + [tokenizer.eos_id]

    # Prompt for assistant response
    ids.append(tokenizer.assistant_id)
    return ids


def chat_loop(model: SpakieGPT, tokenizer: SpakieTokenizer, config: SpakieConfig,
              device: torch.device, system_msg: str = DEFAULT_SYSTEM,
              temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
              json_mode: bool = False):
    """Interactive chat REPL."""
    history = []
    print(f"\nSpakie Chat (type 'quit' to exit, 'clear' to reset)")
    print(f"System: {system_msg}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt_ids = build_prompt_ids(tokenizer, history, system_msg)

        # Truncate history if too long
        while len(prompt_ids) > config.max_seq_len - 64 and len(history) > 1:
            history.pop(0)
            prompt_ids = build_prompt_ids(tokenizer, history, system_msg)

        if json_mode:
            response_text = generate_json(
                model, tokenizer, prompt_ids,
                temperature=temperature, top_k=top_k, top_p=top_p, device=device,
            )
            if response_text is None:
                response_text = "(failed to generate valid JSON)"
        else:
            response_ids = generate(
                model, tokenizer, prompt_ids,
                temperature=temperature, top_k=top_k, top_p=top_p, device=device,
            )
            response_text = tokenizer.decode(response_ids)

        print(f"Spakie: {response_text}\n")
        history.append({"role": "assistant", "content": response_text})
