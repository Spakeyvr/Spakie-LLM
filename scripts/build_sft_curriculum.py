"""Blend targeted SFT examples with a deterministic broad-retention sample."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_number}: missing messages array")
            rows.append(row)
    return rows


def conversation_key(row: dict) -> str:
    return json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build broad-plus-targeted SFT curriculum")
    parser.add_argument("--base", default="data/chat/train.jsonl")
    parser.add_argument("--target", required=True)
    parser.add_argument("--base-sample", type=int, default=1000)
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=[],
        help="Exclude exact conversations found in this JSONL (repeatable)",
    )
    parser.add_argument(
        "--exclude-target-jsonl",
        action="append",
        default=[],
        help="Remove target conversations already found in this JSONL (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base = read_jsonl(args.base)
    target = read_jsonl(args.target)
    excluded_target_keys = {
        conversation_key(row)
        for path in args.exclude_target_jsonl
        for row in read_jsonl(path)
    }
    if excluded_target_keys:
        before = len(target)
        target = [row for row in target if conversation_key(row) not in excluded_target_keys]
        print(f"Excluded from target: {before - len(target)}")
    excluded_keys = {
        conversation_key(row)
        for path in args.exclude_jsonl
        for row in read_jsonl(path)
    }
    if excluded_keys:
        before = len(base)
        base = [row for row in base if conversation_key(row) not in excluded_keys]
        print(f"Excluded from broad base: {before - len(base)}")
    if args.base_sample < 0:
        raise ValueError("--base-sample must be non-negative")
    sampled_base = rng.sample(base, min(args.base_sample, len(base)))

    combined: list[dict] = []
    seen: set[str] = set()
    for row in target + sampled_base:
        key = conversation_key(row)
        if key not in seen:
            seen.add(key)
            combined.append(row)
    rng.shuffle(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tmp_path = args.output + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, args.output)
    print(f"Target examples: {len(target)}")
    print(f"Broad retention sample: {len(sampled_base)}")
    print(f"Deduplicated curriculum: {len(combined)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while building SFT curriculum.")
        sys.exit(130)
