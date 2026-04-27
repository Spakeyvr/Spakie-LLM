"""Preprocess raw corpora into tokenized .npy files for pretraining.

Supports:
- markdown/plain-text files already present in data/raw
- JSONL shards created by scripts/download_pretrain_corpus.py
- streaming token shard creation to avoid one giant in-memory token list
- document dedup, lightweight quality filters, dry-run estimation, and reports
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig, normalize_corpus_source
from tokenizer.train_tokenizer import SpakieTokenizer


SUPPORTED_EXTENSIONS = (".md", ".txt", ".jsonl")
IGNORED_FILENAMES = {"progress.json"}
IGNORED_PATTERNS = ("seen_", ".manifest.")
DEFAULT_TOKENIZE_BATCH_SIZE = 512
DEFAULT_TOKENIZE_BATCH_CHARS = 8_000_000
MAX_RECOMMENDED_TOKENIZER_THREADS = 16


@dataclass
class DocumentRecord:
    source: str
    path: str
    text: str
    metadata: dict


@dataclass
class PendingTokenization:
    source: str
    text: str


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_hash(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_clean_text_for_hash(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_source(root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return normalize_corpus_source(path.parent.name or "unknown")
    parts = rel.parts
    if parts and parts[0] == "large_corpus" and len(parts) > 1:
        return normalize_corpus_source(parts[1])
    if parts:
        return normalize_corpus_source(parts[0])
    return normalize_corpus_source("unknown")


def iter_input_files(raw_root: Path, source_glob: str | None, source_dirs: list[str] | None) -> list[Path]:
    files: list[Path] = []
    if source_dirs:
        for directory in source_dirs:
            base = Path(directory)
            if not base.is_absolute():
                base = raw_root / base
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(base.rglob(f"*{ext}"))
    elif source_glob:
        files = [Path(path) for path in glob.glob(source_glob, recursive=True)]
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(raw_root.rglob(f"*{ext}"))
    unique_files = sorted({
        path.resolve()
        for path in files
        if path.is_file()
        and path.name not in IGNORED_FILENAMES
        and not any(pattern in path.name for pattern in IGNORED_PATTERNS)
    })
    return [Path(path) for path in unique_files]


def iter_documents(raw_root: Path, files: list[Path]) -> Iterable[DocumentRecord]:
    for path in files:
        source = infer_source(raw_root, path)
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = payload.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    metadata = payload.get("meta", {})
                    metadata["line_number"] = line_number
                    yield DocumentRecord(source=source, path=str(path), text=text, metadata=metadata)
        else:
            with path.open("r", encoding="utf-8") as handle:
                yield DocumentRecord(source=source, path=str(path), text=handle.read(), metadata={})


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for line in lines:
        counts[line] += 1
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(lines), 1)


def noise_ratio(text: str) -> float:
    if not text:
        return 1.0
    noisy = sum(not (ch.isalnum() or ch.isspace() or ch in ".,;:!?-_'\"()[]{}") for ch in text)
    return noisy / len(text)


def looks_boilerplate_heavy(text: str) -> bool:
    lower = text.lower()
    markers = (
        "privacy policy", "terms of service", "cookie policy", "all rights reserved",
        "sign in", "subscribe", "javascript required", "skip to content",
    )
    return sum(marker in lower for marker in markers) >= 2


def pick_token_dtype(tokenizer: SpakieTokenizer):
    return np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32


def min_doc_chars_for_source(config: SpakieConfig, source: str) -> int:
    return config.source_min_doc_chars.get(source, config.min_doc_chars)


def recommended_tokenizer_threads(cpu_count: int | None = None) -> int:
    cores = cpu_count or os.cpu_count() or 1
    if cores <= 4:
        return max(1, cores)
    return max(1, min(MAX_RECOMMENDED_TOKENIZER_THREADS, cores - 2))


class TokenShardWriter:
    def __init__(self, shard_dir: Path, shard_size: int, dtype):
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.dtype = dtype
        self.buffer = np.empty(shard_size, dtype=dtype)
        self.offset = 0
        self.index = 0
        self.paths: list[Path] = []

    def add(self, token_ids: list[int]) -> None:
        token_array = np.asarray(token_ids, dtype=self.dtype)
        start = 0
        total = int(token_array.shape[0])
        while start < total:
            room = self.shard_size - self.offset
            take = min(room, total - start)
            end = start + take
            self.buffer[self.offset:self.offset + take] = token_array[start:end]
            self.offset += take
            start = end
            if self.offset == self.shard_size:
                self._flush_full()

    def _write_shard(self, shard_tokens: np.ndarray) -> None:
        path = self.shard_dir / f"tokens-{self.index:05d}.npy"
        np.save(path, shard_tokens)
        self.paths.append(path)
        self.index += 1

    def _flush_full(self) -> None:
        self._write_shard(self.buffer)
        self.offset = 0

    def close(self) -> list[Path]:
        if self.offset:
            self._write_shard(self.buffer[:self.offset].copy())
            self.offset = 0
        return self.paths


def merge_shards(shard_paths: list[Path], train_path: Path, val_path: Path, train_fraction: float, dtype) -> tuple[int, int]:
    total_tokens = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths)
    split_idx = int(total_tokens * train_fraction)

    train_arr = np.lib.format.open_memmap(train_path, mode="w+", dtype=dtype, shape=(split_idx,))
    val_arr = np.lib.format.open_memmap(val_path, mode="w+", dtype=dtype, shape=(total_tokens - split_idx,))

    cursor = 0
    train_cursor = 0
    val_cursor = 0
    for path in shard_paths:
        shard = np.load(path, mmap_mode="r")
        shard_len = int(shard.shape[0])
        shard_start = cursor
        shard_end = cursor + shard_len

        train_take = max(0, min(split_idx, shard_end) - shard_start)
        if train_take:
            train_arr[train_cursor:train_cursor + train_take] = shard[:train_take]
            train_cursor += train_take
        if train_take < shard_len:
            val_slice = shard[train_take:]
            val_arr[val_cursor:val_cursor + len(val_slice)] = val_slice
            val_cursor += len(val_slice)
        cursor = shard_end

    train_arr.flush()
    val_arr.flush()
    return split_idx, total_tokens - split_idx


def should_keep_document(text: str, config: SpakieConfig, source: str) -> tuple[bool, str]:
    if len(text) < min_doc_chars_for_source(config, source):
        return False, "too_short"
    if repeated_line_ratio(text) > config.max_repeated_line_ratio:
        return False, "repeated_lines"
    if noise_ratio(text) > config.max_noise_ratio:
        return False, "too_noisy"
    if looks_boilerplate_heavy(text):
        return False, "boilerplate"
    return True, "kept"


def build_report(
    config: SpakieConfig,
    files: list[Path],
    raw_bytes: int,
    target_tokens: int,
    total_tokens: int,
    source_plan: dict[str, dict[str, int | str | bool]],
    source_stats: dict[str, dict],
    dry_run: bool,
) -> dict:
    estimated_tokens = sum(int(stats["chars_kept"] / config.estimated_chars_per_token) for stats in source_stats.values())
    return {
        "target_train_tokens": config.target_train_tokens,
        "target_processed_tokens": target_tokens,
        "discovered_files": len(files),
        "discovered_raw_bytes": raw_bytes,
        "processed_tokens": total_tokens,
        "estimated_tokens_from_chars": estimated_tokens,
        "gap_to_target": max(target_tokens - total_tokens, 0),
        "dry_run": dry_run,
        "source_targets": source_plan,
        "source_stats": source_stats,
    }


def prepare_data(
    config: SpakieConfig | None = None,
    *,
    target_tokens: int | None = None,
    target_train_tokens: int | None = None,
    dedup: bool = True,
    report_path: str | None = None,
    source_glob: str | None = None,
    source_dirs: list[str] | None = None,
    dry_run: bool = False,
    tokenizer_threads: int | None = None,
    tokenize_batch_size: int = DEFAULT_TOKENIZE_BATCH_SIZE,
    tokenize_batch_chars: int = DEFAULT_TOKENIZE_BATCH_CHARS,
) -> dict:
    config = config or SpakieConfig()
    if target_train_tokens and target_train_tokens > 0:
        config.target_train_tokens = target_train_tokens
        config.pretrain_target_tokens = target_train_tokens
        config.refresh_derived_fields()
    target_tokens = target_tokens or config.target_processed_tokens
    source_plan = config.scaled_corpus_source_plan(target_processed_tokens=target_tokens)
    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    token_dtype = pick_token_dtype(tokenizer)
    tokenizer_threads = tokenizer_threads or recommended_tokenizer_threads()
    tokenize_batch_size = max(1, tokenize_batch_size)
    tokenize_batch_chars = max(1, tokenize_batch_chars)

    raw_root = Path(config.raw_data_dir).resolve()
    files = iter_input_files(raw_root, source_glob=source_glob, source_dirs=source_dirs)
    if not files:
        raise FileNotFoundError(f"No supported files found in {raw_root}")

    raw_bytes = sum(path.stat().st_size for path in files)
    print(f"Found {len(files):,} input files")
    print(f"Discovered raw bytes: {raw_bytes:,}")

    source_stats: dict[str, dict] = defaultdict(lambda: {
        "documents_seen": 0,
        "documents_kept": 0,
        "documents_dropped": 0,
        "drop_reasons": defaultdict(int),
        "chars_kept": 0,
        "raw_bytes": 0,
        "tokens_kept": 0,
    })
    seen_hashes: set[str] = set()

    total_tokens = 0
    shard_paths: list[Path] = []
    shard_dir = Path(config.token_shard_dir)
    if shard_dir.exists() and not dry_run:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    writer = TokenShardWriter(shard_dir, shard_size=config.token_shard_size, dtype=token_dtype)
    pending: list[PendingTokenization] = []
    pending_chars = 0

    def flush_pending() -> bool:
        nonlocal pending, pending_chars, total_tokens
        if not pending:
            return False

        texts = [item.text for item in pending]
        if tokenizer_threads == 1 or len(texts) == 1:
            encoded_batch = [tokenizer.encode(text) + [tokenizer.eos_id] for text in texts]
        else:
            encoded_batch = tokenizer.encode_batch(texts, add_eos=True, num_threads=tokenizer_threads)

        reached_target = False
        for item, token_ids in zip(pending, encoded_batch):
            stats = source_stats[item.source]
            source_cap = int(source_plan.get(item.source, {}).get("target_tokens", 0))
            if source_cap and stats["tokens_kept"] >= source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue
            if dedup:
                doc_hash = hashlib.sha1(normalize_clean_text_for_hash(item.text).encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    stats["documents_dropped"] += 1
                    stats["drop_reasons"]["duplicate"] += 1
                    continue
                seen_hashes.add(doc_hash)
            stats["documents_kept"] += 1
            stats["chars_kept"] += len(item.text)
            if source_cap and stats["tokens_kept"] + len(token_ids) > source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue
            stats["tokens_kept"] += len(token_ids)
            total_tokens += len(token_ids)
            writer.add(token_ids)
            if total_tokens >= target_tokens:
                print(f"Reached target token budget: {total_tokens:,}")
                reached_target = True
                break

        pending = []
        pending_chars = 0
        return reached_target

    if not dry_run:
        print(
            "Tokenizing with "
            f"{tokenizer_threads} thread(s), batches up to "
            f"{tokenize_batch_size:,} docs / {tokenize_batch_chars:,} chars"
        )

    for doc in iter_documents(raw_root, files):
        stats = source_stats[doc.source]
        stats["documents_seen"] += 1
        stats["raw_bytes"] += len(doc.text.encode("utf-8"))

        text = clean_text(doc.text)
        source_cap = int(source_plan.get(doc.source, {}).get("target_tokens", 0))
        if source_cap and stats["tokens_kept"] >= source_cap:
            stats["documents_dropped"] += 1
            stats["drop_reasons"]["source_cap_reached"] += 1
            continue

        keep, reason = should_keep_document(text, config, doc.source)
        if not keep:
            stats["documents_dropped"] += 1
            stats["drop_reasons"][reason] += 1
            continue

        if dedup and dry_run:
            doc_hash = hashlib.sha1(normalize_clean_text_for_hash(text).encode("utf-8")).hexdigest()
            if doc_hash in seen_hashes:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["duplicate"] += 1
                continue
            seen_hashes.add(doc_hash)

        estimated_tokens = max(1, int(len(text) / config.estimated_chars_per_token))
        if dry_run:
            stats["documents_kept"] += 1
            stats["chars_kept"] += len(text)
            if source_cap and stats["tokens_kept"] + estimated_tokens > source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue
            stats["tokens_kept"] += estimated_tokens
            total_tokens += estimated_tokens
        else:
            pending.append(PendingTokenization(source=doc.source, text=text))
            pending_chars += len(text)
            if len(pending) >= tokenize_batch_size or pending_chars >= tokenize_batch_chars:
                if flush_pending():
                    break

        if total_tokens >= target_tokens:
            print(f"Reached target token budget: {total_tokens:,}")
            break

    if not dry_run:
        flush_pending()
        shard_paths = writer.close()

    normalized_source_stats = {}
    for source, stats in source_stats.items():
        source_target = source_plan.get(source, {})
        target_tokens_for_source = int(source_target.get("target_tokens", 0))
        normalized_source_stats[source] = {
            "documents_seen": stats["documents_seen"],
            "documents_kept": stats["documents_kept"],
            "documents_dropped": stats["documents_dropped"],
            "drop_reasons": dict(stats["drop_reasons"]),
            "chars_kept": stats["chars_kept"],
            "raw_bytes": stats["raw_bytes"],
            "tokens_kept": stats["tokens_kept"],
            "target_tokens": target_tokens_for_source,
            "target_raw_chars": int(source_target.get("target_raw_chars", 0)),
            "kind": str(source_target.get("kind", "unplanned")),
            "completion_ratio": (
                stats["tokens_kept"] / target_tokens_for_source
                if target_tokens_for_source
                else 0.0
            ),
        }

    for source, source_target in source_plan.items():
        normalized_source_stats.setdefault(source, {
            "documents_seen": 0,
            "documents_kept": 0,
            "documents_dropped": 0,
            "drop_reasons": {},
            "chars_kept": 0,
            "raw_bytes": 0,
            "tokens_kept": 0,
            "target_tokens": int(source_target.get("target_tokens", 0)),
            "target_raw_chars": int(source_target.get("target_raw_chars", 0)),
            "kind": str(source_target.get("kind", "unknown")),
            "completion_ratio": 0.0,
        })

    report = build_report(
        config,
        files,
        raw_bytes,
        target_tokens,
        total_tokens,
        source_plan,
        normalized_source_stats,
        dry_run=dry_run,
    )
    report["target_tokens_requested"] = target_tokens

    processed_dir = Path(config.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dest = Path(report_path or config.corpus_report_path)
    report_dest.parent.mkdir(parents=True, exist_ok=True)
    with report_dest.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if dry_run:
        print(f"Estimated processed tokens: {total_tokens:,}")
        print(f"Gap to target: {max(target_tokens - total_tokens, 0):,}")
        return report

    if not shard_paths:
        raise RuntimeError("No token shards were produced")

    output_dtype = np.uint16 if token_dtype == np.uint16 else np.uint32
    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    if output_dtype != np.uint16:
        raise ValueError("The current training stack expects uint16-compatible token ids")
    train_tokens, val_tokens = merge_shards(shard_paths, train_path, val_path, config.train_split_fraction, output_dtype)

    report["processed_tokens"] = train_tokens + val_tokens
    report["train_tokens"] = train_tokens
    report["val_tokens"] = val_tokens
    report["gap_to_target"] = max(target_tokens - report["processed_tokens"], 0)
    with report_dest.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Train: {train_tokens:,} tokens -> {train_path}")
    print(f"Val:   {val_tokens:,} tokens -> {val_path}")
    print(f"Report: {report_dest}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a large text corpus for pretraining")
    parser.add_argument("--target_tokens", type=int, default=0, help="Stop once this many processed tokens are reached")
    parser.add_argument("--target_train_tokens", type=int, default=0, help="Derived processed target from desired train tokens")
    parser.add_argument("--dedup", dest="dedup", action="store_true", help="Enable document-level deduplication")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="Disable document-level deduplication")
    parser.set_defaults(dedup=True)
    parser.add_argument("--report_path", type=str, default="", help="Where to write the corpus report JSON")
    parser.add_argument("--source_glob", type=str, default="", help="Optional file glob for inputs")
    parser.add_argument("--source_dirs", type=str, default="", help="Comma-separated input directories relative to data/raw")
    parser.add_argument("--dry_run", action="store_true", help="Estimate token totals without writing train/val arrays")
    parser.add_argument(
        "--tokenizer_threads",
        type=int,
        default=0,
        help="SentencePiece tokenizer threads for prepare tokenization (0 = recommended auto)",
    )
    parser.add_argument(
        "--tokenize_batch_size",
        type=int,
        default=DEFAULT_TOKENIZE_BATCH_SIZE,
        help="Maximum documents per tokenizer batch",
    )
    parser.add_argument(
        "--tokenize_batch_chars",
        type=int,
        default=DEFAULT_TOKENIZE_BATCH_CHARS,
        help="Maximum cleaned characters per tokenizer batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = [part.strip() for part in args.source_dirs.split(",") if part.strip()] or None
    prepare_data(
        target_tokens=args.target_tokens or None,
        target_train_tokens=args.target_train_tokens or None,
        dedup=args.dedup,
        report_path=args.report_path or None,
        source_glob=args.source_glob or None,
        source_dirs=source_dirs,
        dry_run=args.dry_run,
        tokenizer_threads=args.tokenizer_threads or None,
        tokenize_batch_size=args.tokenize_batch_size,
        tokenize_batch_chars=args.tokenize_batch_chars,
    )


if __name__ == "__main__":
    main()
