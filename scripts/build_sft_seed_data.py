"""Write permanent, clearly named local SFT seed sources.

The merge step reads these files like any other source. Keeping them as JSONL
instead of injecting them in memory makes the exact identity and repair data
easy to inspect, version, and reproduce.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig
from scripts.prepare_sft import (
    ANTI_ECHO_SEEDS,
    _FACTUAL_REPAIR_SEEDS,
    build_assistant_seed_examples,
    build_identity_seed_examples,
    build_pair_seed_examples,
    deduplicate_examples,
)


def write_source(path: str, source_name: str, examples: list[dict]) -> int:
    deduped, _ = deduplicate_examples(examples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for example in deduped:
            row = {"source": source_name, "messages": example["messages"]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)
    return len(deduped)


def main() -> int:
    output_dir = SpakieConfig().chat_raw_dir
    sources = {
        "spakie_180m_identity": build_identity_seed_examples(None),
        "assistant_behavior": build_assistant_seed_examples(None, repeats=1),
        "anti_echo": build_pair_seed_examples(ANTI_ECHO_SEEDS, None, repeats=1),
        "factual_repairs": build_pair_seed_examples(_FACTUAL_REPAIR_SEEDS, None, repeats=1),
    }
    for source_name, examples in sources.items():
        path = os.path.join(output_dir, f"{source_name}.jsonl")
        count = write_source(path, source_name, examples)
        print(f"{source_name}: {count:,} examples -> {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted while writing permanent SFT seed files.")
        raise SystemExit(130)
