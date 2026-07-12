"""Export the full local SmolTalk pool as a descriptive raw-chat batch."""

from __future__ import annotations

import json
import os
import sys


SOURCE = "data/chat_raw/smoltalk.jsonl"
OUTPUT = "data/raw_chat/general_assistant_instruction_qa_smoltalk_expansion.jsonl"


def main() -> None:
    rows: list[dict] = []
    seen: set[str] = set()
    with open(SOURCE, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{SOURCE}:{line_number}: missing messages")
            key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append({"messages": messages})

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTPUT)
    print(f"{len(rows):,} {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while exporting general-assistant data.")
        sys.exit(130)
