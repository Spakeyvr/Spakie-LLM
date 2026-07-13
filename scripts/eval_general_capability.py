"""Evaluate one checkpoint on a broad, held-out general-capability suite.

The suite is external JSONL so prompts stay fixed across SFT iterations. Outputs
are written incrementally and survive Ctrl+C, making long local evaluations
auditable and resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_basic_qa import answer_question_mlx, resolve_checkpoint


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


def score_answer(row: dict[str, Any], answer: str) -> tuple[bool | None, str]:
    check = row.get("check", {})
    kind = check.get("type", "manual")
    value = check.get("value")
    text = normalized(answer)
    if kind == "manual":
        return None, "manual review"
    if kind == "contains_all":
        missing = [item for item in value if normalized(str(item)) not in text]
        return not missing, "missing: " + ", ".join(missing) if missing else "all required terms present"
    if kind == "contains_any":
        found = [item for item in value if normalized(str(item)) in text]
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
        return not missing, "valid JSON" if not missing else "missing keys: " + ", ".join(missing)
    raise ValueError(f"unknown check type {kind!r} for {row['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate broad general capability")
    parser.add_argument("--preset", default="180m")
    parser.add_argument("--checkpoint", default="sft_best.safetensors")
    parser.add_argument("--prompts", default="data/eval/general_capability.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from configs.default import get_preset_config, inherit_attention_shape_from_tensors, inherit_mlp_shape_from_tensors
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
    flat = load_safetensors(checkpoint)
    model_flat = {key[len("model."):]: value for key, value in flat.items() if key.startswith("model.")}
    saved_config = load_mlx_checkpoint_config(checkpoint, allow_legacy_config=False)
    config = saved_config or inherit_mlp_shape_from_tensors(
        inherit_attention_shape_from_tensors(config, model_flat), model_flat
    )
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
    counts: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    manual: Counter[str] = Counter()
    try:
        with open(args.output, mode, encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                if row["id"] in completed:
                    continue
                answer = answer_question_mlx(model, tokenizer, config, row["prompt"], "")
                result, reason = score_answer(row, answer)
                record = {**row, "checkpoint": checkpoint, "answer": answer, "passed": result, "score_reason": reason}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                counts[row["category"]] += 1
                if result is True:
                    passed[row["category"]] += 1
                elif result is None:
                    manual[row["category"]] += 1
                print(f"[{index}/{len(rows)}] {row['id']}: {result} | {answer[:100]!r}")
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results are saved to {args.output}.")
        sys.exit(130)

    scored_total = sum(counts.values()) - sum(manual.values())
    passed_total = sum(passed.values())
    print(f"Scored: {passed_total}/{scored_total}; manual review: {sum(manual.values())}")
    for category in sorted(counts):
        scored = counts[category] - manual[category]
        print(f"  {category}: {passed[category]}/{scored} scored, {manual[category]} manual")


if __name__ == "__main__":
    main()
