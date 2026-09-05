"""CLI chat REPL."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import SpakieConfig
from model.transformer import SpakieGPT
from runtime import RuntimeSettings
from tokenizer.train_tokenizer import SpakieTokenizer
from inference.continuation import decode_prefilled_continuation
from inference.generate import generate, generate_json
from inference.generation_utils import prompt_token_budget


def build_prompt_ids(tokenizer: SpakieTokenizer, history: list[dict], system_msg: str) -> list[int]:
    """Build token IDs from conversation history using chat template."""
    return tokenizer.apply_chat_template(history, system_msg=system_msg, add_assistant_prompt=True)


def chat_loop(model: SpakieGPT, tokenizer: SpakieTokenizer, config: SpakieConfig,
              runtime: RuntimeSettings, system_msg: str = "",
              temperature: float = 0.1, top_k: int = 1, top_p: float = 1.0,
              json_mode: bool = False, max_new_tokens: int = 256,
              repetition_penalty: float = 1.2):
    """Interactive chat REPL."""
    history = []
    print(f"\nSpakie Chat (type '/quit' to exit, '/clear' to reset)")
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
        prompt_ids = build_prompt_ids(tokenizer, history, system_msg)

        # Reserve the actual requested response budget, not a hard-coded guess.
        prompt_budget = prompt_token_budget(config.max_seq_len, max_new_tokens)
        while len(prompt_ids) > prompt_budget and len(history) > 1:
            history.pop(0)
            prompt_ids = build_prompt_ids(tokenizer, history, system_msg)

        if json_mode:
            response_text = generate_json(
                model, tokenizer, prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p, runtime=runtime,
            )
            if response_text is None:
                response_text = "(failed to generate valid JSON)"
        else:
            response_ids = generate(
                model, tokenizer, prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                runtime=runtime,
            )
            response_text = tokenizer.decode(response_ids)

        if not response_text.strip():
            # Do not carry an invalid empty assistant turn into the next prompt.
            history.pop()
            print("Spakie: (no response generated; turn was not added to history)\n")
            continue
        print(f"Spakie: {response_text}\n")
        history.append({"role": "assistant", "content": response_text})


def continuation_loop(model: SpakieGPT, tokenizer: SpakieTokenizer, config: SpakieConfig,
                      runtime: RuntimeSettings, temperature: float = 0.8,
                      top_k: int = 50, top_p: float = 0.9,
                      max_new_tokens: int = 256,
                      repetition_penalty: float = 1.0,
                      show_special_tokens: bool = False):
    """Interactive raw text continuation REPL for pretrained checkpoints."""
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
            model, tokenizer, prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            runtime=runtime,
            stop_on_special_tokens=False,
            ban_special_tokens=False,
        )
        completion_text = decode_prefilled_continuation(
            tokenizer,
            prompt_ids,
            response_ids,
            show_special_tokens=show_special_tokens,
        )
        print(f"{completion_text}\n")
