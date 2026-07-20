"""Correct known systematic labels in the structured-instruction curriculum."""

from __future__ import annotations

import argparse
import json
import os
import sys


CORRECTIONS = {
    "Answer only YES or NO: Is 35 divisible by 3?": "NO",
    "Reply with exactly 3 words describing a cold morning. Do not add punctuation.": "Cold crisp silent",
}

GENERIC_PREFIXES = (
    "Task: ",
    "Please answer carefully. ",
    "Follow the requested format exactly. ",
    "Provide the direct answer. ",
    "Solve this request accurately: ",
)


def unwrap_prompt(prompt: str) -> str:
    for prefix in GENERIC_PREFIXES:
        if prompt.startswith(prefix):
            return prompt[len(prefix) :]
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    temp_path = args.output + ".tmp"
    correction_counts = {prompt: 0 for prompt in CORRECTIONS}
    rows = 0
    try:
        with open(args.input, "r", encoding="utf-8") as source, open(
            temp_path, "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{args.input}:{line_number} has invalid messages")
                user_prompt = unwrap_prompt(str(messages[0].get("content", "")))
                if user_prompt in CORRECTIONS:
                    messages[-1]["content"] = CORRECTIONS[user_prompt]
                    correction_counts[user_prompt] += 1
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1

        unexpected = {
            prompt: count for prompt, count in correction_counts.items() if count != 6
        }
        if unexpected:
            raise ValueError(f"expected six wrapper variants per correction, got {unexpected}")
        os.replace(temp_path, args.output)
        print(f"Wrote {rows:,} corrected rows to {args.output}")
        for prompt, count in correction_counts.items():
            print(f"Corrected {count} rows: {prompt}")
    except KeyboardInterrupt:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        print("\nInterrupted; no partial dataset was published.")
        sys.exit(130)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


if __name__ == "__main__":
    main()
