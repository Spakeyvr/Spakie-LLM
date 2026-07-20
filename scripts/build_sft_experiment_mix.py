"""Build a deterministic weighted SFT mixture with evaluation-leakage guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path


def read_jsonl(path: str, *, require_messages: bool = True) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if require_messages and not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_number} has no messages list")
            rows.append(row)
    return rows


def normalized(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


GENERIC_PROMPT_PREFIXES = (
    "task: ",
    "please answer carefully. ",
    "follow the requested format exactly. ",
    "provide the direct answer. ",
    "solve this request accurately: ",
)


def canonical_user_prompt(text: object) -> str:
    """Normalize a prompt and remove generic wrappers used by local curricula."""
    prompt = normalized(text)
    changed = True
    while changed:
        changed = False
        for prefix in GENERIC_PROMPT_PREFIXES:
            if prompt.startswith(prefix):
                prompt = prompt[len(prefix):].strip()
                changed = True
                break
    return prompt


def user_prompts(row: dict) -> set[str]:
    return {
        canonical_user_prompt(message.get("content", ""))
        for message in row.get("messages", [])
        if message.get("role") == "user"
    }


def conversation_signature(row: dict) -> str:
    messages = [
        {
            "role": message.get("role"),
            "content": normalized(message.get("content", "")),
            **({"train": False} if message.get("train") is False else {}),
        }
        for message in row.get("messages", [])
    ]
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def parse_target_spec(spec: str) -> tuple[str, int]:
    try:
        path, repeat_text = spec.rsplit("=", 1)
        repeat = int(repeat_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("target must use PATH=REPEAT") from exc
    if repeat <= 0:
        raise argparse.ArgumentTypeError("target repeat must be positive")
    return path, repeat


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mix(
    baseline_rows: list[dict],
    targets: list[tuple[str, list[dict], int]],
    *,
    baseline_sample: int,
    seed: int,
    blocked_prompts: set[str],
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    if baseline_sample > 0 and baseline_sample < len(baseline_rows):
        baseline_indices = rng.sample(range(len(baseline_rows)), baseline_sample)
        selected_baseline = [baseline_rows[index] for index in baseline_indices]
    else:
        selected_baseline = list(baseline_rows)

    combined = list(selected_baseline)
    baseline_signatures = {conversation_signature(row) for row in baseline_rows}
    target_report: list[dict] = []
    leakage_dropped = 0
    seen_target: set[str] = set()
    for path, rows, repeat in targets:
        accepted: list[dict] = []
        duplicates_with_baseline = 0
        duplicates_with_targets = 0
        for row in rows:
            if user_prompts(row) & blocked_prompts:
                leakage_dropped += 1
                continue
            signature = conversation_signature(row)
            if signature in seen_target:
                duplicates_with_targets += 1
                continue
            seen_target.add(signature)
            if signature in baseline_signatures:
                duplicates_with_baseline += 1
            accepted.append(row)
        for replica in range(repeat):
            for row in accepted:
                weighted = dict(row)
                weighted["experiment_source_file"] = os.path.basename(path)
                weighted["experiment_weight_replica"] = replica + 1
                combined.append(weighted)
        target_report.append(
            {
                "path": path,
                "input_rows": len(rows),
                "unique_accepted_rows": len(accepted),
                "repeat": repeat,
                "weighted_rows": len(accepted) * repeat,
                "duplicates_with_baseline": duplicates_with_baseline,
                "duplicates_with_other_targets": duplicates_with_targets,
            }
        )

    rng.shuffle(combined)
    return combined, {
        "baseline_input_rows": len(baseline_rows),
        "baseline_selected_rows": len(selected_baseline),
        "targets": target_report,
        "evaluation_leakage_rows_dropped": leakage_dropped,
        "output_rows": len(combined),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        type=parse_target_spec,
        metavar="PATH=REPEAT",
    )
    parser.add_argument("--baseline-sample", type=int, default=0, help="0 keeps the full baseline")
    parser.add_argument("--eval-prompts", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    raw_root = Path("data/chat_raw").resolve()
    if raw_root not in output.parents:
        parser.error("--output must be inside data/chat_raw/")

    tmp_path = str(output) + ".tmp"
    try:
        baseline_rows = read_jsonl(args.baseline)
        targets = [(path, read_jsonl(path), repeat) for path, repeat in args.target]
        blocked_prompts = {
            canonical_user_prompt(row["prompt"])
            for path in args.eval_prompts
            for row in read_jsonl(path, require_messages=False)
            if "prompt" in row
        }
        rows, report = build_mix(
            baseline_rows,
            targets,
            baseline_sample=args.baseline_sample,
            seed=args.seed,
            blocked_prompts=blocked_prompts,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output)
        report.update(
            {
                "schema_version": 1,
                "baseline": args.baseline,
                "baseline_sha256": sha256_file(args.baseline),
                "evaluation_prompt_files": args.eval_prompts,
                "output": str(output),
                "output_sha256": sha256_file(str(output)),
            }
        )
        manifest_path = str(output) + ".manifest.json"
        with open(manifest_path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(manifest_path + ".tmp", manifest_path)
        print(f"Wrote {len(rows):,} rows to {output}")
        print(f"Manifest: {manifest_path}")
    except KeyboardInterrupt:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print("\nInterrupted; no partial mixture was published.")
        sys.exit(130)


if __name__ == "__main__":
    main()
