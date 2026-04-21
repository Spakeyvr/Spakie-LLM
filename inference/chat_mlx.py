"""CLI chat REPL (MLX backend)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import SpakieConfig
from inference.generate_mlx import generate, generate_json
from model.transformer_mlx import SpakieGPTMLX
from tokenizer.train_tokenizer import SpakieTokenizer


DEFAULT_SYSTEM = "Answer clearly and factually. Keep explanations simple, direct, and truthful."


def _build_prompt_ids(tokenizer: SpakieTokenizer, history: list[dict], system_msg: str) -> list[int]:
    return tokenizer.apply_chat_template(history, system_msg=system_msg, add_assistant_prompt=True)


def chat_loop(
    model: SpakieGPTMLX,
    tokenizer: SpakieTokenizer,
    config: SpakieConfig,
    system_msg: str = DEFAULT_SYSTEM,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    json_mode: bool = False,
):
    history: list[dict] = []
    print("\nSpakie Chat (type '/quit' to exit, '/clear' to reset)")
    if system_msg:
        print(f"System: {system_msg}\n")
    else:
        print("System: (none)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Bye!")
            break
        if user_input.lower() == "/clear":
            history.clear()
            print("(conversation cleared)\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt_ids = _build_prompt_ids(tokenizer, history, system_msg)
        while len(prompt_ids) > config.max_seq_len - 64 and len(history) > 1:
            history.pop(0)
            prompt_ids = _build_prompt_ids(tokenizer, history, system_msg)

        if json_mode:
            response_text = generate_json(
                model, tokenizer, prompt_ids,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            if response_text is None:
                response_text = "(failed to generate valid JSON)"
        else:
            response_ids = generate(
                model, tokenizer, prompt_ids,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            response_text = tokenizer.decode(response_ids)

        print(f"Spakie: {response_text}\n")
        history.append({"role": "assistant", "content": response_text})
