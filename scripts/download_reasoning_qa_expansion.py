"""Download reasoning-oriented QA batches into the required raw-chat directory."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_sft_data import load_arc, load_openbookqa, write_jsonl


def main() -> None:
    output_dir = "data/raw_chat"
    os.makedirs(output_dir, exist_ok=True)
    sources = (
        (
            "qa_science_reasoning_arc_challenge.jsonl",
            load_arc("ARC-Challenge", 10_000, 42, "Challenge"),
        ),
        (
            "qa_science_commonsense_openbookqa.jsonl",
            load_openbookqa(10_000, 42),
        ),
    )
    for name, rows in sources:
        path = os.path.join(output_dir, name)
        write_jsonl(path, rows)
        print(f"{len(rows):,} {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while downloading reasoning QA data.")
        sys.exit(130)
