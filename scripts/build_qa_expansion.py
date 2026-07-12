"""Export cleaned local QA corpora into descriptive raw-chat batches."""

from __future__ import annotations

import json
import os
import sys


SOURCES = (
    ("data/chat_raw/squad.jsonl", "data/raw_chat/qa_reading_comprehension_squad.jsonl"),
    ("data/chat_raw/boolq.jsonl", "data/raw_chat/qa_boolean_context_reasoning_boolq.jsonl"),
)


def export_source(source: str, output: str) -> None:
    rows: list[dict] = []
    seen: set[str] = set()
    with open(source, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"{source}:{line_number}: expected one user/assistant pair")
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError(f"{source}:{line_number}: invalid role order")
            if not all(isinstance(message.get("content"), str) and message["content"].strip() for message in messages):
                continue
            cleaned = {"messages": messages}
            key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(cleaned)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    tmp = output + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, output)
    print(f"{len(rows):,} {output}")


def main() -> None:
    for source, output in SOURCES:
        export_source(source, output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while exporting QA data.")
        sys.exit(130)
