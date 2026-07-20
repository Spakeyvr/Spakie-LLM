"""Rescore saved capability outputs against the current fixed prompt contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_general_capability import read_jsonl, score_answer, sha256_file, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--prompts", default="data/eval/general_capability.jsonl")
    parser.add_argument("--output", required=True, help="Summary JSON path")
    parser.add_argument("--rescored-output", default="", help="Optional rescored JSONL path")
    parser.add_argument("--exclude-id", action="append", default=[], help="Prompt id to exclude")
    parser.add_argument("--manual-ratings", default="", help="Optional JSONL with id, passed, and reason")
    args = parser.parse_args()

    excluded = set(args.exclude_id)
    prompts = [row for row in read_jsonl(args.prompts) if row["id"] not in excluded]
    results = [row for row in read_jsonl(args.results) if row["id"] not in excluded]
    manual_ratings = {}
    if args.manual_ratings:
        manual_ratings = {row["id"]: row for row in read_jsonl(args.manual_ratings)}
    prompt_by_id = {row["id"]: row for row in prompts}
    if len(prompt_by_id) != len(prompts):
        raise ValueError(f"duplicate prompt ids in {args.prompts}")

    result_ids = {row["id"] for row in results}
    expected_ids = set(prompt_by_id)
    missing = expected_ids - result_ids
    extra = result_ids - expected_ids
    if missing or extra or len(results) != len(prompts):
        raise ValueError(
            f"result/prompt mismatch: missing={sorted(missing)}, extra={sorted(extra)}, "
            f"results={len(results)}, prompts={len(prompts)}"
        )

    rescored: list[dict] = []
    for result in results:
        contract = prompt_by_id[result["id"]]
        answer = result.get("answer", "")
        passed, reason = score_answer(contract, answer)
        manual_rating = manual_ratings.get(result["id"])
        if manual_rating is not None:
            passed = bool(manual_rating["passed"])
            reason = str(manual_rating.get("reason", "manual rubric rating"))
        rescored.append(
            {
                **contract,
                "checkpoint": result.get("checkpoint", "historical checkpoint unavailable"),
                "answer": answer,
                "passed": passed,
                "score_reason": reason,
                **({"manual_passed": passed} if manual_rating is not None else {}),
                "generation": result.get("generation", {
                    "max_new_tokens": 96,
                    "temperature": 0.1,
                    "top_k": 1,
                    "top_p": 1.0,
                    "repetition_penalty": 1.2,
                    "system_prompt": "",
                }),
            }
        )

    summary = summarize_results(rescored)
    summary.update(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results": args.results,
            "results_sha256": sha256_file(args.results),
            "prompts": args.prompts,
            "prompts_sha256": sha256_file(args.prompts),
            "checkpoint": rescored[0].get("checkpoint", "") if rescored else "",
            "excluded_ids": sorted(excluded),
            "manual_ratings": args.manual_ratings,
        }
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(args.output + ".tmp", args.output)

    if args.rescored_output:
        os.makedirs(os.path.dirname(args.rescored_output) or ".", exist_ok=True)
        with open(args.rescored_output + ".tmp", "w", encoding="utf-8") as handle:
            for row in rescored:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(args.rescored_output + ".tmp", args.rescored_output)

    print(
        f"Scored {summary['passed']}/{summary['scored']} | "
        f"micro={summary['micro_accuracy']:.4f} macro={summary['macro_accuracy']:.4f}"
    )
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; no partial summary was published.")
        sys.exit(130)
