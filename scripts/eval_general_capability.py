"""Evaluate one checkpoint on a broad, held-out general-capability suite.

The suite is external JSONL so prompts stay fixed across SFT iterations. Outputs
are written incrementally and survive Ctrl+C, making long local evaluations
auditable and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_basic_qa import resolve_checkpoint


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"row-{line_number}")
            rows.append(row)
    return rows


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_term(text: str, term: object) -> bool:
    """Match words/numbers on token boundaries and symbols as literal text."""
    haystack = normalized(text)
    needle = normalized(str(term))
    if not needle:
        return False
    if re.search(r"[a-z0-9]", needle):
        return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None
    return needle in haystack


def score_answer(row: dict[str, Any], answer: str) -> tuple[bool | None, str]:
    check = row.get("check", {})
    kind = check.get("type", "manual")
    value = check.get("value")
    text = normalized(answer)
    rejected = [item for item in check.get("reject_any", []) if contains_term(answer, item)]
    if rejected:
        return False, "rejected contradiction: " + ", ".join(map(str, rejected))
    if kind == "all_of":
        reasons: list[str] = []
        for index, subcheck in enumerate(value or [], start=1):
            passed, reason = score_answer({**row, "check": subcheck}, answer)
            reasons.append(f"{index}: {reason}")
            if passed is not True:
                return passed, "failed composite check; " + "; ".join(reasons)
        return True, "all composite checks passed; " + "; ".join(reasons)
    if kind == "manual":
        return None, "manual review"
    if kind == "contains_all":
        missing = [item for item in value if not contains_term(answer, item)]
        return not missing, "missing: " + ", ".join(missing) if missing else "all required terms present"
    if kind == "contains_any":
        found = [item for item in value if contains_term(answer, item)]
        return bool(found), "matched: " + ", ".join(found) if found else "no accepted term present"
    if kind == "exact":
        accepted = value if isinstance(value, list) else [value]
        passed = text in {normalized(str(item)) for item in accepted}
        return passed, "exact match" if passed else "expected exact answer"
    if kind == "regex":
        passed = re.search(str(value), answer, flags=re.IGNORECASE | re.MULTILINE) is not None
        return passed, "regex matched" if passed else f"regex did not match: {value}"
    if kind == "json":
        try:
            parsed = json.loads(answer.strip())
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        required = check.get("required_keys", [])
        if not isinstance(parsed, dict):
            return False, "JSON root is not an object"
        missing = [key for key in required if key not in parsed]
        if missing:
            return False, "missing keys: " + ", ".join(missing)
        expected = check.get("expected", {})
        wrong = [key for key, expected_value in expected.items() if parsed.get(key) != expected_value]
        return not wrong, "valid JSON" if not wrong else "wrong values for: " + ", ".join(wrong)
    if kind == "exact_json":
        try:
            parsed = json.loads(answer.strip())
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc.msg}"
        passed = parsed == value
        return passed, "exact JSON match" if passed else "JSON value did not match"
    if kind == "choice":
        letter = str(check["letter"]).upper()
        answer_terms = check.get("answer_terms", [])
        starts_with_choice = re.search(
            rf"^\s*(?:answer\s*[:=-]\s*)?{re.escape(letter)}(?:\s*[.),\]:-]|\s+)",
            answer,
            flags=re.IGNORECASE,
        ) is not None
        missing_terms = [term for term in answer_terms if not contains_term(answer, term)]
        if not starts_with_choice:
            return False, f"answer does not start with choice {letter}"
        return not missing_terms, (
            "choice and answer text matched"
            if not missing_terms
            else "missing answer terms: " + ", ".join(map(str, missing_terms))
        )
    if kind == "line_set":
        expected = {normalized(item) for item in value}
        actual = {normalized(line) for line in answer.splitlines() if line.strip()}
        passed = actual == expected and len([line for line in answer.splitlines() if line.strip()]) == len(expected)
        return passed, "exact line set" if passed else f"expected lines {sorted(expected)}, got {sorted(actual)}"
    raise ValueError(f"unknown check type {kind!r} for {row['id']}")


def answer_question_mlx(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    from inference.chat_mlx import _build_prompt_ids
    from inference.generate_mlx import generate

    prompt_ids = _build_prompt_ids(tokenizer, [{"role": "user", "content": prompt}], "")
    response_ids = generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(response_ids).strip()


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    manual: Counter[str] = Counter()
    for row in rows:
        if row.get("manual_passed") is not None:
            result = bool(row["manual_passed"])
        else:
            result, _ = score_answer(row, row.get("answer", ""))
        category = row["category"]
        counts[category] += 1
        if result is True:
            passed[category] += 1
        elif result is None:
            manual[category] += 1

    categories: dict[str, dict[str, float | int]] = {}
    for category in sorted(counts):
        scored = counts[category] - manual[category]
        categories[category] = {
            "passed": passed[category],
            "scored": scored,
            "manual": manual[category],
            "accuracy": passed[category] / scored if scored else 0.0,
        }
    scored_total = sum(item["scored"] for item in categories.values())
    passed_total = sum(item["passed"] for item in categories.values())
    macro_accuracy = (
        sum(float(item["accuracy"]) for item in categories.values()) / len(categories)
        if categories
        else 0.0
    )
    return {
        "passed": passed_total,
        "scored": scored_total,
        "manual": sum(item["manual"] for item in categories.values()),
        "micro_accuracy": passed_total / scored_total if scored_total else 0.0,
        "macro_accuracy": macro_accuracy,
        "categories": categories,
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broad general capability")
    parser.add_argument("--preset", default="180m")
    parser.add_argument("--checkpoint", default="sft_best.safetensors")
    parser.add_argument("--prompts", default="data/eval/general_capability.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--summary", default="", help="Summary JSON path (default: <output>.summary.json)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from configs.default import get_preset_config
    from model.transformer_mlx import SpakieGPTMLX
    from runtime.checkpoint_io import (
        load_mlx_checkpoint_config,
        load_mlx_checkpoint_meta,
        load_mlx_model_weights_strict,
        validate_checkpoint_tokenizer,
    )
    from runtime.mlx_backend import load_safetensors, resolve_mlx_runtime
    from tokenizer.train_tokenizer import SpakieTokenizer

    config = get_preset_config(args.preset)
    checkpoint = resolve_checkpoint(config, args.checkpoint, "mlx")
    config = load_mlx_checkpoint_config(checkpoint)
    flat = load_safetensors(checkpoint)
    model_flat = {key[len("model."):]: value for key, value in flat.items() if key.startswith("model.")}
    validate_checkpoint_tokenizer(
        load_mlx_checkpoint_meta(checkpoint),
        config.tokenizer_prefix + ".model",
        source=checkpoint,
    )
    model = SpakieGPTMLX(config)
    load_mlx_model_weights_strict(model, flat, path=checkpoint)
    del flat, model_flat
    runtime = resolve_mlx_runtime(args.precision)
    if runtime.dtype is not None:
        model.set_dtype(runtime.dtype)
    model.eval()
    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")

    completed: set[str] = set()
    mode = "w"
    if args.resume and os.path.exists(args.output):
        completed = {row["id"] for row in read_jsonl(args.output)}
        mode = "a"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = read_jsonl(args.prompts)
    try:
        with open(args.output, mode, encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                if row["id"] in completed:
                    continue
                answer = answer_question_mlx(
                    model,
                    tokenizer,
                    row["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                result, reason = score_answer(row, answer)
                record = {
                    **row,
                    "checkpoint": checkpoint,
                    "answer": answer,
                    "passed": result,
                    "score_reason": reason,
                    "generation": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "system_prompt": "",
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{index}/{len(rows)}] {row['id']}: {result} | {answer[:100]!r}")
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results are saved to {args.output}.")
        sys.exit(130)

    result_rows = read_jsonl(args.output)
    summary = summarize_results(result_rows)
    summary.update(
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256_file(checkpoint),
            "prompts": args.prompts,
            "prompts_sha256": sha256_file(args.prompts),
            "output": args.output,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "system_prompt": "",
            },
        }
    )
    summary_path = args.summary or args.output + ".summary.json"
    with open(summary_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(summary_path + ".tmp", summary_path)

    print(
        f"Scored: {summary['passed']}/{summary['scored']}; manual review: {summary['manual']} | "
        f"micro={summary['micro_accuracy']:.4f} macro={summary['macro_accuracy']:.4f}"
    )
    for category, item in summary["categories"].items():
        print(f"  {category}: {item['passed']}/{item['scored']} scored, {item['manual']} manual")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed prompt outputs remain saved.")
        sys.exit(130)
