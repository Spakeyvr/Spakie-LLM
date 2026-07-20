"""Summarize strict prompt accuracy and atomic instruction compliance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_general_capability import read_jsonl, score_answer, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = read_jsonl(args.prompts)
    results = read_jsonl(args.results)
    result_by_id = {row["id"]: row for row in results}
    if set(result_by_id) != {row["id"] for row in prompts}:
        raise ValueError("result IDs do not match benchmark prompt IDs")

    prompt_passed: Counter[str] = Counter()
    prompt_total: Counter[str] = Counter()
    unit_passed: Counter[str] = Counter()
    unit_total: Counter[str] = Counter()
    detail: list[dict] = []
    for prompt in prompts:
        answer = result_by_id[prompt["id"]].get("answer", "")
        category = prompt["category"]
        strict, strict_reason = score_answer(prompt, answer)
        prompt_total[category] += 1
        if strict is True:
            prompt_passed[category] += 1
        units: list[dict] = []
        for unit in prompt.get("unit_checks", []):
            passed, reason = score_answer({**prompt, "check": unit["check"]}, answer)
            unit_total[category] += 1
            if passed is True:
                unit_passed[category] += 1
            units.append({"name": unit["name"], "passed": passed is True, "reason": reason})
        detail.append({
            "id": prompt["id"],
            "category": category,
            "strict_passed": strict is True,
            "strict_reason": strict_reason,
            "units": units,
        })

    categories = {}
    for category in sorted(prompt_total):
        categories[category] = {
            "prompt_passed": prompt_passed[category],
            "prompt_total": prompt_total[category],
            "prompt_accuracy": prompt_passed[category] / prompt_total[category],
            "unit_passed": unit_passed[category],
            "unit_total": unit_total[category],
            "unit_accuracy": unit_passed[category] / unit_total[category] if unit_total[category] else 0.0,
        }
    total_prompts = sum(prompt_total.values())
    total_prompt_passed = sum(prompt_passed.values())
    total_units = sum(unit_total.values())
    total_unit_passed = sum(unit_passed.values())
    summary = {
        "schema_version": 1,
        "prompts": args.prompts,
        "prompts_sha256": sha256_file(args.prompts),
        "results": args.results,
        "results_sha256": sha256_file(args.results),
        "strict_prompt_passed": total_prompt_passed,
        "strict_prompt_total": total_prompts,
        "strict_prompt_accuracy": total_prompt_passed / total_prompts if total_prompts else 0.0,
        "unit_passed": total_unit_passed,
        "unit_total": total_units,
        "unit_accuracy": total_unit_passed / total_units if total_units else 0.0,
        "categories": categories,
        "detail": detail,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(output)
    print(
        f"Strict prompts: {total_prompt_passed}/{total_prompts}; "
        f"instruction units: {total_unit_passed}/{total_units}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no partial benchmark summary was published.")
        raise SystemExit(130)
