"""Download SFT source datasets into per-source JSONL files under data/chat_raw/.

Each source writes its own file (e.g. data/chat_raw/alpaca.jsonl) containing
only user/assistant message pairs — no system prompt. Merging, deduplication,
and system-prompt injection happen later in scripts/prepare_sft.py, which also
picks up any custom JSONL files you drop into data/chat_raw/ yourself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


def trim(text: str) -> str:
    return " ".join(str(text).split()).strip()


def make_example(user_text: str, assistant_text: str) -> dict | None:
    user_text = trim(user_text)
    assistant_text = trim(assistant_text)
    if not user_text or not assistant_text:
        return None
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def take_rows(dataset, limit: int, seed: int):
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def clean_chat_messages(messages: object, *, fold_system_into_user: bool = False) -> list[dict]:
    if not isinstance(messages, list):
        return []

    cleaned = []
    system_parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            return []
        role = msg.get("role")
        content = trim(msg.get("content", ""))
        if role == "system":
            if fold_system_into_user and content:
                system_parts.append(content)
            continue
        if role not in ("user", "assistant") or not content:
            return []
        if cleaned and cleaned[-1]["role"] == role:
            return []
        if role == "user" and system_parts:
            system_text = "\n\n".join(system_parts)
            content = f"{system_text}\n\n{content}"
            system_parts = []
        cleaned.append({"role": role, "content": content})

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return []
    return cleaned


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


def load_no_robots(limit: int, seed: int) -> list[dict]:
    print("Loading No Robots...")
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")
    rows = take_rows(dataset, limit, seed)
    examples = []
    for row in rows:
        cleaned = clean_chat_messages(row.get("messages"))
        if cleaned:
            examples.append({"messages": cleaned})
    return examples


def load_smoltalk(limit: int, seed: int) -> list[dict]:
    print("Loading SmolTalk...")
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    if limit > 0:
        rows = dataset.shuffle(seed=seed, buffer_size=10_000).take(limit)
    else:
        rows = dataset

    examples = []
    for row in rows:
        cleaned = clean_chat_messages(row.get("messages"), fold_system_into_user=True)
        if cleaned:
            examples.append({"messages": cleaned})
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


def write_jsonl(path: str, examples: list[dict]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def main() -> None:
    config = SpakieConfig()
    parser = argparse.ArgumentParser(description="Download SFT sources to data/chat_raw/")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for per-source caps")
    parser.add_argument(
        "--sources",
        type=str,
        default="alpaca,dolly,no_robots,smoltalk,squad,sciq,boolq,arc_easy,arc_challenge,openbookqa",
        help="Comma-separated SFT sources to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.chat_raw_dir,
        help=f"Directory for per-source raw JSONL files (default: {config.chat_raw_dir})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    source_builders = {
        "alpaca": lambda limit: download_alpaca(limit, args.seed),
        "dolly": lambda limit: load_dolly(limit, args.seed),
        "no_robots": lambda limit: load_no_robots(limit, args.seed),
        "smoltalk": lambda limit: load_smoltalk(limit, args.seed),
        "squad": lambda limit: load_squad(limit, args.seed),
        "sciq": lambda limit: load_sciq(limit, args.seed),
        "boolq": lambda limit: load_boolq(limit, args.seed),
        "arc_easy": lambda limit: load_arc("ARC-Easy", limit, args.seed, "Easy"),
        "arc_challenge": lambda limit: load_arc("ARC-Challenge", limit, args.seed, "Challenge"),
        "openbookqa": lambda limit: load_openbookqa(limit, args.seed),
    }
    requested_sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    unknown_sources = sorted(set(requested_sources) - set(source_builders))
    if unknown_sources:
        print(f"Unknown sources: {', '.join(unknown_sources)}")
        sys.exit(2)

    for source_name in requested_sources:
        builder = source_builders[source_name]
        limit = config.sft_source_limits.get(source_name, 0)
        try:
            examples = builder(limit)
        except Exception as exc:
            print(f"  skipping {source_name}: {exc}")
            continue
        out_path = os.path.join(args.output_dir, f"{source_name}.jsonl")
        write_jsonl(out_path, examples)
        print(f"  {source_name}: {len(examples):,} examples -> {out_path}")

    print(f"\nRaw SFT files written to {args.output_dir}/.")
    print("Next: run `python3 scripts/prepare_sft.py` to merge into data/chat/train.jsonl.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while downloading SFT data.")
        sys.exit(130)
