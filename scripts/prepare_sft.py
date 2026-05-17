"""Merge SFT JSONL files from data/chat_raw/ into a single training file.

Each *.jsonl file in the raw dir is treated as one source — the filename stem
is the source name, used to look up an optional per-source cap from
`SpakieConfig.sft_source_limits`. Any custom JSONL you drop into data/chat_raw/
is picked up automatically; sources without a cap entry are taken in full.

The chosen system prompt is injected into every example. If the raw example
already carries a system message, it is replaced (or stripped when
--no-system is set).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import CHAT_SYSTEM_PROMPT, SpakieConfig


# Templated prefixes injected by scripts/download_sft_data.py (squad, sciq, boolq, arc, openbookqa).
# Matched only at the start of a user message so we never eat legitimate content.
_TEMPLATE_PREFIX_RE = re.compile(r"^\s*(?:Question|Q)\s*:\s*", re.IGNORECASE)
_TEMPLATE_BLOCK_RE = re.compile(
    r"(?:\n+\s*|\s+)(?:Context|Reference|Passage|Choices)\s*:\s*",
    re.IGNORECASE,
)
_TEMPLATE_TRAILER_RE = re.compile(
    r"\n*\s*(?:Answer\s+(?:yes\s+or\s+no\s+)?(?:clearly|using\s+the\s+context|the\s+question)\.?|"
    r"Select\s+the\s+correct\s+answer\.?)\s*$",
    re.IGNORECASE,
)

DISALLOWED_SFT_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<tools>",
    "</tools>",
    "Action:",
    "Observation:",
    "Final Answer:",
    "You are an expert in composing functions",
)


def strip_question_template(content: str) -> str:
    """Strip 'Question: ... Context: ... Answer clearly.' scaffolding from a user turn.

    Reorders to '<context>\n\n<question>' so the model still learns context-aware Q&A
    without learning that 'Question:' is a required trigger word.
    """
    if not _TEMPLATE_PREFIX_RE.match(content):
        return content

    body = _TEMPLATE_PREFIX_RE.sub("", content, count=1)
    body = _TEMPLATE_TRAILER_RE.sub("", body)

    match = _TEMPLATE_BLOCK_RE.search(body)
    if match is None:
        return body.strip()

    question = body[: match.start()].strip()
    context = body[match.end():].strip()
    if not question or not context:
        return body.strip()
    return f"{context}\n\n{question}"


def contains_disallowed_sft_marker(messages: object) -> bool:
    """Return True when any message content contains tool/reasoning scaffolding."""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and any(marker in content for marker in DISALLOWED_SFT_MARKERS):
            return True
    return False


def normalize_example(raw: dict, system_prompt: str | None) -> dict | None:
    """Return a clean {messages: [...]} dict or None if malformed.

    system_prompt=None means strip any system message; a string (empty allowed)
    means prepend exactly one system message with that content.
    """
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return None

    cleaned: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        if role == "system":
            continue
        content = content.strip()
        if role == "user":
            content = strip_question_template(content)
        if not content:
            return None
        cleaned.append({"role": role, "content": content})

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return None

    if system_prompt is not None:
        cleaned.insert(0, {"role": "system", "content": system_prompt})

    return {"messages": cleaned}


def signature(example: dict) -> tuple:
    # Dedup on user/assistant content only — the system prompt is uniform.
    return tuple(
        (msg["role"], msg["content"])
        for msg in example["messages"]
        if msg["role"] != "system"
    )


def load_source(path: str, system_prompt: str | None, limit: int, seed: int) -> list[dict]:
    examples: list[dict] = []
    malformed = 0
    filtered = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if contains_disallowed_sft_marker(raw.get("messages")):
                filtered += 1
                continue
            example = normalize_example(raw, system_prompt)
            if example is None:
                malformed += 1
                continue
            examples.append(example)
    if malformed:
        print(f"    warning: skipped {malformed} malformed lines in {os.path.basename(path)}")
    if filtered:
        print(f"    filtered {filtered} tool/template artifact examples in {os.path.basename(path)}")
    if limit > 0 and len(examples) > limit:
        random.Random(seed).shuffle(examples)
        examples = examples[:limit]
    return examples


def main() -> None:
    config = SpakieConfig()
    parser = argparse.ArgumentParser(description="Merge data/chat_raw/*.jsonl into data/chat/train.jsonl")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=config.chat_raw_dir,
        help=f"Directory of per-source JSONL files (default: {config.chat_raw_dir})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(config.chat_data_dir, "train.jsonl"),
        help="Output path for the merged JSONL",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=CHAT_SYSTEM_PROMPT,
        help="System prompt to inject into every example",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not include any system message (overrides --system)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Global cap on total merged examples (0 = no cap)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Optional comma-separated source names (filename stems) to include; default = all *.jsonl",
    )
    args = parser.parse_args()

    if args.no_system:
        system_prompt: str | None = None
        system_label = "(no system message)"
    else:
        system_prompt = args.system
        system_label = "(empty system message)" if system_prompt == "" else repr(system_prompt)

    if not os.path.isdir(args.input_dir):
        print(f"Input dir not found: {args.input_dir}")
        print("Run `python3 scripts/download_sft_data.py` first, or drop your own *.jsonl files in there.")
        sys.exit(2)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    if not paths:
        print(f"No *.jsonl files found in {args.input_dir}/.")
        sys.exit(2)

    requested = {s.strip() for s in args.sources.split(",") if s.strip()}
    if requested:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in requested]
        missing = requested - {os.path.splitext(os.path.basename(p))[0] for p in paths}
        if missing:
            print(f"Missing sources in {args.input_dir}/: {', '.join(sorted(missing))}")
            sys.exit(2)

    print(f"System prompt: {system_label}")

    all_examples: list[dict] = []
    counts: Counter[str] = Counter()
    for path in paths:
        source_name = os.path.splitext(os.path.basename(path))[0]
        limit = config.sft_source_limits.get(source_name, 0)
        examples = load_source(path, system_prompt, limit, args.seed)
        counts[source_name] = len(examples)
        cap_note = f" (cap {limit:,})" if limit > 0 else ""
        print(f"  {source_name}: {len(examples):,} examples{cap_note}")
        all_examples.extend(examples)

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for example in all_examples:
        sig = signature(example)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(example)
    dropped = len(all_examples) - len(deduped)
    if dropped:
        print(f"Removed {dropped:,} duplicate examples")

    random.Random(args.seed).shuffle(deduped)
    if args.max > 0 and len(deduped) > args.max:
        deduped = deduped[: args.max]
        print(f"Capped output at {args.max:,} examples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_path = args.output + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for example in deduped:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    os.replace(tmp_path, args.output)

    print(f"\nSaved {len(deduped):,} examples to {args.output}")
    for source_name in sorted(counts):
        print(f"  {source_name}: {counts[source_name]:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while preparing SFT data.")
        sys.exit(130)
