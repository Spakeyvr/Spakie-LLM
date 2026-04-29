"""Build a mixed SFT dataset geared toward factual QA and simple helpful dialogue.

The default mix intentionally favors question answering and grounded
explanations over generic instruction-following so small local models can
answer basic questions more reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


SYSTEM_PROMPT = "Answer clearly and factually. Keep explanations simple, direct, and truthful."


def trim(text: str) -> str:
    return " ".join(str(text).split()).strip()


def make_example(user_text: str, assistant_text: str, *, include_system: bool = True) -> dict | None:
    user_text = trim(user_text)
    assistant_text = trim(assistant_text)
    if not user_text or not assistant_text:
        return None

    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})
    return {"messages": messages}


def take_rows(dataset, limit: int, seed: int):
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def download_alpaca(limit: int, seed: int) -> list[dict]:
    print("Loading Alpaca Clean...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        instruction = trim(row.get("instruction", ""))
        inp = trim(row.get("input", ""))
        output = trim(row.get("output", ""))
        if not instruction or not output:
            continue
        user_text = f"{instruction}\n\nContext: {inp}" if inp else instruction
        example = make_example(user_text, output)
        if example is not None:
            examples.append(example)
    return examples


def load_dolly(limit: int, seed: int) -> list[dict]:
    print("Loading Dolly...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        instruction = trim(row.get("instruction", ""))
        context = trim(row.get("context", ""))
        response = trim(row.get("response", ""))
        if not instruction or not response:
            continue
        user_text = f"{instruction}\n\nContext: {context}" if context else instruction
        example = make_example(user_text, response)
        if example is not None:
            examples.append(example)
    return examples


def load_squad(limit: int, seed: int) -> list[dict]:
    print("Loading SQuAD...")
    dataset = load_dataset("squad", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        question = trim(row.get("question", ""))
        context = trim(row.get("context", ""))
        answers = row.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        answer = trim(answer_texts[0]) if answer_texts else ""
        user_text = f"Question: {question}\n\nContext: {context}\n\nAnswer using the context."
        example = make_example(user_text, answer)
        if example is not None:
            examples.append(example)
    return examples


def load_sciq(limit: int, seed: int) -> list[dict]:
    print("Loading SciQ...")
    dataset = load_dataset("allenai/sciq", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        question = trim(row.get("question", ""))
        support = trim(row.get("support", ""))
        answer = trim(row.get("correct_answer", ""))
        user_text = f"Question: {question}\n\nReference: {support}\n\nAnswer clearly."
        example = make_example(user_text, answer)
        if example is not None:
            examples.append(example)
    return examples


def load_boolq(limit: int, seed: int) -> list[dict]:
    print("Loading BoolQ...")
    dataset = load_dataset("google/boolq", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        question = trim(row.get("question", ""))
        passage = trim(row.get("passage", ""))
        answer = "Yes." if bool(row.get("answer")) else "No."
        user_text = f"Question: {question}\n\nContext: {passage}\n\nAnswer yes or no using the context."
        example = make_example(user_text, answer)
        if example is not None:
            examples.append(example)
    return examples


def format_choices(choices: dict | None) -> str:
    if not isinstance(choices, dict):
        return ""
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = []
    for label, text in zip(labels, texts, strict=False):
        label = trim(label)
        text = trim(text)
        if label and text:
            pairs.append(f"{label}. {text}")
    return "\n".join(pairs)


def load_arc(name: str, limit: int, seed: int, label: str) -> list[dict]:
    print(f"Loading ARC {label}...")
    dataset = load_dataset("allenai/ai2_arc", name, split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        question = trim(row.get("question", ""))
        choices_text = format_choices(row.get("choices"))
        answer_key = trim(row.get("answerKey", ""))
        if not question or not choices_text or not answer_key:
            continue
        answer_text = ""
        choices = row.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts = choices.get("text", []) if isinstance(choices, dict) else []
        for option_label, option_text in zip(labels, texts, strict=False):
            if trim(option_label) == answer_key:
                answer_text = trim(option_text)
                break
        assistant = f"{answer_key}. {answer_text}" if answer_text else answer_key
        user_text = f"Question: {question}\n\nChoices:\n{choices_text}\n\nSelect the correct answer."
        example = make_example(user_text, assistant)
        if example is not None:
            examples.append(example)
    return examples


def load_openbookqa(limit: int, seed: int) -> list[dict]:
    print("Loading OpenBookQA...")
    dataset = load_dataset("allenai/openbookqa", "main", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        question = trim(row.get("question_stem", ""))
        choices_text = format_choices(row.get("choices"))
        answer_key = trim(row.get("answerKey", ""))
        if not question or not choices_text or not answer_key:
            continue
        answer_text = ""
        choices = row.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts = choices.get("text", []) if isinstance(choices, dict) else []
        for option_label, option_text in zip(labels, texts, strict=False):
            if trim(option_label) == answer_key:
                answer_text = trim(option_text)
                break
        assistant = f"{answer_key}. {answer_text}" if answer_text else answer_key
        user_text = f"Question: {question}\n\nChoices:\n{choices_text}\n\nSelect the correct answer."
        example = make_example(user_text, assistant)
        if example is not None:
            examples.append(example)
    return examples


def dedup_examples(examples: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for example in examples:
        signature = tuple((msg["role"], msg["content"]) for msg in example["messages"])
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(example)
    return unique


def main() -> None:
    config = SpakieConfig()
    parser = argparse.ArgumentParser(description="Build the canonical SFT dataset")
    parser.add_argument(
        "--max",
        type=int,
        default=config.sft_download_max_examples,
        help=f"Max total examples (0 = all, default: {config.sft_download_max_examples})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--output-name",
        type=str,
        default="train.jsonl",
        help="Output filename inside data/chat/",
    )
    args = parser.parse_args()

    os.makedirs(config.chat_data_dir, exist_ok=True)

    source_builders = [
        ("alpaca", lambda limit: download_alpaca(limit, args.seed)),
        # ("dolly", lambda limit: load_dolly(limit, args.seed)),
        # ("squad", lambda limit: load_squad(limit, args.seed)),
        # ("sciq", lambda limit: load_sciq(limit, args.seed)),
        # ("boolq", lambda limit: load_boolq(limit, args.seed)),
        # ("arc_easy", lambda limit: load_arc("ARC-Easy", limit, args.seed, "Easy")),
        # ("arc_challenge", lambda limit: load_arc("ARC-Challenge", limit, args.seed, "Challenge")),
        # ("openbookqa", lambda limit: load_openbookqa(limit, args.seed)),
    ]

    all_examples = []
    counts = Counter()
    for source_name, builder in source_builders:
        limit = config.sft_source_limits.get(source_name, 0)
        try:
            examples = builder(limit)
            all_examples.extend(examples)
            counts[source_name] = len(examples)
            print(f"  {source_name}: {len(examples):,} examples")
        except Exception as exc:
            print(f"  skipping {source_name}: {exc}")

    all_examples = dedup_examples(all_examples)
    random.Random(args.seed).shuffle(all_examples)

    if args.max > 0:
        all_examples = all_examples[:args.max]

    out_path = os.path.join(config.chat_data_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as handle:
        for example in all_examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_examples):,} examples to {out_path}")
    for source_name in sorted(counts):
        print(f"  {source_name}: {counts[source_name]:,}")


if __name__ == "__main__":
    main()
