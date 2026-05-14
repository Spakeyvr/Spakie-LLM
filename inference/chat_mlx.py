"""CLI chat REPL (MLX backend)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import CHAT_SYSTEM_PROMPT, SpakieConfig
from inference.generate_mlx import generate, generate_json
from model.transformer_mlx import SpakieGPTMLX
from tokenizer.train_tokenizer import SpakieTokenizer


DEFAULT_SYSTEM = CHAT_SYSTEM_PROMPT


def _build_prompt_ids(tokenizer: SpakieTokenizer, history: list[dict], system_msg: str) -> list[int]:
    return tokenizer.apply_chat_template(history, system_msg=system_msg, add_assistant_prompt=True)


def chat_loop(
    model: SpakieGPTMLX,
    tokenizer: SpakieTokenizer,
    config: SpakieConfig,
    system_msg: str = DEFAULT_SYSTEM,
    temperature: float = 0.1,
    top_k: int = 1,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.2,
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
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            response_text = tokenizer.decode(response_ids)

        print(f"Spakie: {response_text}\n")
        history.append({"role": "assistant", "content": response_text})


def continuation_loop(
    model: SpakieGPTMLX,
    tokenizer: SpakieTokenizer,
    config: SpakieConfig,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    repetition_penalty: float = 1.0,
    show_special_tokens: bool = False,
):
    print("\nSpakie Continuation (type '/quit' to exit)")

    while True:
        try:
            prompt = input("Text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt.strip():
            continue
        if prompt.strip().lower() == "/quit":
            print("Bye!")
            break

        prompt_ids = tokenizer.encode(prompt)
        prompt_ids = prompt_ids[-max(1, config.max_seq_len - max_new_tokens):]
        response_ids = generate(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_special_tokens=False,
            ban_special_tokens=False,
        )
        response_text = tokenizer.decode(response_ids, skip_special_tokens=not show_special_tokens)
        print(f"{response_text}\n")
