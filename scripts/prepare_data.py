"""Preprocess raw corpora into tokenized .npy files for pretraining.

Supports:
- markdown/plain-text files already present in data/raw
- JSONL shards created by scripts/download_pretrain_corpus.py
- streaming token shard creation to avoid one giant in-memory token list
- document dedup, lightweight quality filters, dry-run estimation, and reports
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import struct
import tempfile
import threading
from array import array
from collections import deque
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import xxhash
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig, normalize_corpus_source
from runtime.langid import is_probably_english, language_id_sample
from runtime.processed_data import (
    invalidate_processed_data,
    publish_processed_data_manifest,
    stable_payload_sha256,
    tokenizer_contract,
    validate_processed_data,
)
from tokenizer.train_tokenizer import SpakieTokenizer


SUPPORTED_EXTENSIONS = (".md", ".txt", ".jsonl")
IGNORED_FILENAMES = {"progress.json"}
IGNORED_PATTERNS = ("seen_", ".manifest.")
DEFAULT_TOKENIZE_BATCH_SIZE = 512
DEFAULT_TOKENIZE_BATCH_CHARS = 8_000_000
MAX_RECOMMENDED_TOKENIZER_THREADS = 16
SHARD_RUN_MANIFEST = "shard_run_manifest.json"
SHARD_RESUME_JOURNAL = "accepted_documents.bin"
PREPARATION_SCHEMA_VERSION = 4
_JOURNAL_HEADER = struct.Struct("<HIIQH")


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
    # When dedup is enabled these are pre-computed by the worker (or serial
    # path) and only used by the main process for fast dedup lookups.
    exact_hash: int | None = None
    signature: np.ndarray | None = None


@dataclass(frozen=True)
class AcceptedDocument:
    source: str
    token_count: int
    char_count: int
    exact_hash: int
    band_keys: tuple[int, ...]


def append_accepted_document(handle, record: AcceptedDocument) -> None:
    source = record.source.encode("utf-8")
    if len(source) > 65_535:
        raise ValueError("Corpus source name is too long for the resume journal")
    handle.write(
        _JOURNAL_HEADER.pack(
            len(source),
            record.token_count,
            record.char_count,
            record.exact_hash,
            len(record.band_keys),
        )
    )
    handle.write(source)
    if record.band_keys:
        handle.write(struct.pack(f"<{len(record.band_keys)}Q", *record.band_keys))


def iter_accepted_documents(path: Path) -> Iterator[AcceptedDocument]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        while True:
            header = handle.read(_JOURNAL_HEADER.size)
            if not header:
                return
            if len(header) != _JOURNAL_HEADER.size:
                raise RuntimeError(f"Truncated resume journal header in {path}")
            source_len, token_count, char_count, exact_hash, band_count = (
                _JOURNAL_HEADER.unpack(header)
            )
            payload = handle.read(source_len + band_count * 8)
            if len(payload) != source_len + band_count * 8:
                raise RuntimeError(f"Truncated resume journal record in {path}")
            source = payload[:source_len].decode("utf-8")
            band_keys = (
                struct.unpack(f"<{band_count}Q", payload[source_len:])
                if band_count
                else ()
            )
            yield AcceptedDocument(
                source=source,
                token_count=int(token_count),
                char_count=int(char_count),
                exact_hash=int(exact_hash),
                band_keys=tuple(int(value) for value in band_keys),
            )


# Lines that are pure navigation chrome — matched whole-line, case-insensitive.
# Kept narrow and literal to avoid stripping legitimate content.
_NAV_LINE_RE = re.compile(
    r"^\s*("
    r"navigation|contents|jump to navigation|jump to search|jump to:"
    r"|from wikipedia,? the free encyclopedia"
    r"|retrieved from"
    r"|this page was last edited"
    r"|navigation menu"
    r"|main page|special pages|permanent link|page information"
    r"|what links here|related changes|upload file"
    r"|print/export|in other projects"
    r"|edit this page|talk|log in|log out|create account"
    r")\s*$",
    re.IGNORECASE,
)

# Wikipedia/citation residue. Numeric references are removed only for sources
# known to be prose; globally deleting ``[1]`` corrupts code indexing and math.
_CITATION_RE = re.compile(
    r"\[(?:edit|citation needed|note\s+\d+|nb\s+\d+|\d{1,3})\]",
    re.IGNORECASE,
)

_CITATION_SOURCES = frozenset({"wikipedia", "wikipedia_snapshot"})
_STRUCTURED_TEXT_SOURCES = frozenset({"python_edu", "openwebmath", "finemath", "arxiv"})
_WEB_RESIDUE_SOURCES = frozenset(
    {"wikipedia", "wikipedia_snapshot", "fineweb", "fineweb-edu"}
)
_KNOWN_HTML_TAG_RE = re.compile(
    r"</?(?:a|article|aside|blockquote|body|br|code|div|em|footer|h[1-6]|head|"
    r"header|html|i|li|main|nav|ol|p|pre|section|span|strong|table|tbody|td|"
    r"th|thead|title|tr|ul)(?:\s+[^<>\n]*)?/?>",
    re.IGNORECASE,
)

# Lines that look like JavaScript/CSS residue from failed HTML extraction.
# Anchored to line start so prose mentioning "function" or "var" is untouched.
_JS_CSS_LINE_RE = re.compile(
    r"^\s*("
    r"function(\s+\w+)?\s*\([^)]*\)\s*\{"
    r"|(?:var|let|const)\s+\w+\s*="
    r"|\};?\s*$"
    r"|@media\s+"
    r"|@import\s+"
    r"|<!--"
    r"|//\s*<!\[CDATA\["
    r")"
)


def clean_text(text: str, source: str = "") -> str:
    """Clean extracted prose without rewriting code or mathematical syntax."""
    text = _KNOWN_HTML_TAG_RE.sub("", text)
    if source in _CITATION_SOURCES:
        text = _CITATION_RE.sub("", text)
    text = re.sub(r"\r\n?", "\n", text)
    if source in _STRUCTURED_TEXT_SOURCES:
        text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    else:
        text = re.sub(r"[ \t]+", " ", text)
    lines = [
        ln for ln in text.splitlines()
        if not _NAV_LINE_RE.match(ln)
        and not (source in _WEB_RESIDUE_SOURCES and _JS_CSS_LINE_RE.match(ln))
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n") if source in _STRUCTURED_TEXT_SOURCES else text.strip()


_SHINGLE_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _iter_shingles(text: str, n: int):
    window: deque[str] = deque(maxlen=max(1, n))
    yielded = False
    for match in _SHINGLE_WORD_RE.finditer(text):
        window.append(match.group(0).lower())
        if len(window) == window.maxlen:
            yielded = True
            yield " ".join(window)
    if window and not yielded:
        yield " ".join(window)


# xxh32 over the shingle bytes — drop-in replacement for datasketch's default
# sha1_hash32, ~3-5x faster and equally well-distributed for MinHash purposes.
def _xxh32_hashfunc(b: bytes) -> int:
    return xxhash.xxh32_intdigest(b)


from functools import lru_cache


@lru_cache(maxsize=None)
def _minhash_params(num_perm: int):
    """datasketch-compatible permutation constants for ``num_perm``.

    Returns ``(a, b, mersenne_prime, max_hash)`` where ``a`` and ``b`` are the
    seeded permutation coefficient arrays a stock ``datasketch.MinHash`` would
    build. Reusing datasketch's own arrays keeps the vectorized signature below
    byte-identical to the per-shingle ``MinHash.update`` path it replaces, so
    the LSH banding downstream sees exactly the same hashvalues.
    """
    from datasketch import MinHash
    import datasketch.minhash as _dm

    ref = MinHash(num_perm=num_perm, hashfunc=_xxh32_hashfunc)
    a, b = ref.permutations
    return (
        np.ascontiguousarray(a, dtype=np.uint64),
        np.ascontiguousarray(b, dtype=np.uint64),
        np.uint64(_dm._mersenne_prime),
        np.uint64(_dm._max_hash),
    )


# Rows of the (shingle x perm) hash matrix processed per numpy block. Keeps the
# temporary bounded (~BLOCK*num_perm*8 bytes) so a single huge document — e.g. a
# full Gutenberg book with ~1M shingles — can't spike a worker's memory.
_MINHASH_BLOCK = 4096


def compute_minhash_signature(
    text: str, *, num_perm: int, shingle_size: int
) -> np.ndarray:
    """Build a MinHash signature for ``text`` and return its raw hashvalues.

    Shingle hashes and the permutation matrix are both processed in bounded
    blocks. Output is byte-identical to the datasketch path without retaining
    every shingle string/hash for multi-megabyte documents.
    """
    a, b, prime, max_hash = _minhash_params(num_perm)
    running = np.full(num_perm, max_hash, dtype=np.uint64)
    block: list[int] = []
    saw_shingle = False

    def reduce_block(values: list[int]) -> None:
        if not values:
            return
        chunk = np.asarray(values, dtype=np.uint64)
        # uint64 arithmetic wraps mod 2^64 elementwise, exactly matching
        # datasketch's scalar (a*hv + b) before the Mersenne modulo.
        phv = ((chunk[:, None] * a[None, :] + b[None, :]) % prime) & max_hash
        np.minimum(running, phv.min(axis=0), out=running)

    for shingle in _iter_shingles(text, max(1, shingle_size)):
        saw_shingle = True
        block.append(xxhash.xxh32_intdigest(shingle.encode("utf-8")))
        if len(block) >= _MINHASH_BLOCK:
            reduce_block(block)
            block.clear()
    reduce_block(block)
    if not saw_shingle:
        return np.full(num_perm, max_hash, dtype=np.uint64)
    return running


def compute_exact_hash(text: str) -> int:
    """xxh64 of the cleaned text, used for byte-identical dedup before MinHash."""
    return xxhash.xxh64_intdigest(text.encode("utf-8"))


class NearDuplicateIndex:
    """MinHash + LSH near-duplicate index with optional exact-hash fast path.

    Catches paraphrased copies and near-identical mirrors that exact-hash
    dedup misses. A document is reported duplicate when its Jaccard similarity
    on word-shingles is above ``threshold`` against any previously inserted
    document.

    Workers compute MinHash signatures + exact hashes in parallel and ship
    them to this index; ``query_signature`` only runs the (cheap) LSH lookup
    on the main process, so the expensive shingling/hashing is no longer the
    serial bottleneck.
    """

    def __init__(self, *, threshold: float, num_perm: int, shingle_size: int):
        # Reuse datasketch's optimal (bands, rows) split so this hand-rolled
        # index bands the signature exactly as MinHashLSH(threshold, num_perm)
        # would (default false-pos/neg weights of 0.5/0.5). We only need
        # bucket *membership* — datasketch's query never verifies Jaccard
        # either — so each band maps to a set of band-key bytes and a hit in
        # any band means "near-duplicate". This drops the per-doc MinHash
        # object construction + LSH query (~120us) to a few plain set lookups.
        from datasketch.lsh import _optimal_param

        bands, rows = _optimal_param(threshold, num_perm, 0.5, 0.5)
        self._ranges: list[tuple[int, int]] = [
            (i * rows, (i + 1) * rows) for i in range(bands)
        ]
        # Store 64-bit fingerprints rather than the full band byte strings.
        # This cuts the dominant dedup index memory by roughly an order of
        # magnitude; the xxh64 collision probability is negligible here.
        self._tables: list[set[int]] = [set() for _ in range(bands)]
        self._exact_hashes: set[int] = set()

    def is_duplicate_exact(self, exact_hash: int) -> bool:
        """Return True if this xxh64 hash has been seen before. Does not insert."""
        return exact_hash in self._exact_hashes

    def query_signature(self, exact_hash: int, signature: np.ndarray) -> bool:
        """Return True if signature collides with a previously inserted doc.

        Inserts on miss. Caller is expected to have already checked the
        exact-hash fast path; we still record the exact hash on miss so a
        later byte-identical doc short-circuits.
        """
        if exact_hash in self._exact_hashes:
            return True
        return self._check_and_insert(exact_hash, signature)

    def _check_and_insert(self, exact_hash: int, signature: np.ndarray) -> bool:
        band_keys = self.band_keys(signature)
        for table, key in zip(self._tables, band_keys):
            if key in table:
                self._exact_hashes.add(exact_hash)
                return True
        for table, key in zip(self._tables, band_keys):
            table.add(key)
        self._exact_hashes.add(exact_hash)
        return False

    def band_keys(self, signature: np.ndarray) -> tuple[int, ...]:
        return tuple(
            xxhash.xxh64_intdigest(signature[start:end].tobytes())
            for start, end in self._ranges
        )

    def insert_known(self, exact_hash: int, band_keys: tuple[int, ...]) -> None:
        if len(band_keys) != len(self._tables):
            raise RuntimeError(
                "Resume journal was produced with a different MinHash band layout"
            )
        self._exact_hashes.add(exact_hash)
        for table, key in zip(self._tables, band_keys):
            table.add(int(key))


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


def _iter_file_documents(raw_root: Path, path: Path, progress: tqdm | None = None) -> Iterable[DocumentRecord]:
    source = infer_source(raw_root, path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if progress is not None:
                    progress.update(len(line.encode("utf-8")))
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
            text = handle.read()
        if progress is not None:
            progress.update(path.stat().st_size)
        yield DocumentRecord(source=source, path=str(path), text=text, metadata={})


def iter_documents(raw_root: Path, files: list[Path], progress: tqdm | None = None) -> Iterable[DocumentRecord]:
    for path in files:
        yield from _iter_file_documents(raw_root, path, progress=progress)


# --- Parallel filtering ---------------------------------------------------
# Workers run clean_text + should_keep_document in separate processes (Python
# regex/string work is GIL-bound, so processes are required for true speedup).
# The serial bits — dedup, source caps, tokenization, shard writes — stay in
# the main process. Pool results are consumed in input-file order. This is an
# important correctness invariant: deduplication and --resume both depend on
# replaying the exact same accepted-document prefix on every run.

_WORKER_CONFIG: SpakieConfig | None = None
_WORKER_RAW_ROOT: Path | None = None
_WORKER_DEDUP_ENABLED: bool = False
_WORKER_NUM_PERM: int = 128
_WORKER_SHINGLE_SIZE: int = 5
_WORKER_RESUME_HASHES = None


def _worker_init(
    config: SpakieConfig,
    raw_root_str: str,
    dedup_enabled: bool,
    num_perm: int,
    shingle_size: int,
    resume_hashes_path: str = "",
) -> None:
    global _WORKER_CONFIG, _WORKER_RAW_ROOT
    global _WORKER_DEDUP_ENABLED, _WORKER_NUM_PERM, _WORKER_SHINGLE_SIZE
    global _WORKER_RESUME_HASHES
    _WORKER_CONFIG = config
    _WORKER_RAW_ROOT = Path(raw_root_str)
    _WORKER_DEDUP_ENABLED = dedup_enabled
    _WORKER_NUM_PERM = num_perm
    _WORKER_SHINGLE_SIZE = shingle_size
    _WORKER_RESUME_HASHES = (
        np.load(resume_hashes_path, mmap_mode="r") if resume_hashes_path else None
    )


def _hash_is_in_sorted_array(exact_hash: int, values) -> bool:
    if values is None or len(values) == 0:
        return False
    index = int(np.searchsorted(values, np.uint64(exact_hash)))
    return index < len(values) and int(values[index]) == exact_hash


def _worker_process_file(file_path_str: str) -> dict:
    if _WORKER_CONFIG is None or _WORKER_RAW_ROOT is None:
        raise RuntimeError("prepare-data worker was not initialized")
    config = _WORKER_CONFIG
    raw_root = _WORKER_RAW_ROOT
    dedup_enabled = _WORKER_DEDUP_ENABLED
    num_perm = _WORKER_NUM_PERM
    shingle_size = _WORKER_SHINGLE_SIZE
    path = Path(file_path_str)
    file_bytes = path.stat().st_size
    documents: list[FilteredDoc] = []
    for doc in _iter_file_documents(raw_root, path):
        raw_bytes = len(doc.text.encode("utf-8"))
        text = clean_text(doc.text, doc.source)
        keep, reason = should_keep_document(text, config, doc.source)
        if keep:
            exact_hash = compute_exact_hash(text)
            signature = None
            if dedup_enabled and not _hash_is_in_sorted_array(
                exact_hash, _WORKER_RESUME_HASHES
            ):
                signature = compute_minhash_signature(
                    text, num_perm=num_perm, shingle_size=shingle_size
                )
        else:
            exact_hash = None
            signature = None
        documents.append((
            doc.source,
            raw_bytes,
            keep,
            None if keep else reason,
            text if keep else None,
            exact_hash,
            signature,
        ))
    return {"file_path": str(path), "file_bytes": file_bytes, "documents": documents}


# Yielded tuple:
#   (source, raw_bytes, kept, drop_reason_or_None, cleaned_text_or_None,
#    exact_hash_or_None, signature_hashvalues_or_None)
FilteredDoc = tuple[
    str, int, bool, "str | None", "str | None", "int | None", "np.ndarray | None"
]


def _doc_stream_serial(
    raw_root: Path,
    files: list[Path],
    config: SpakieConfig,
    progress: tqdm | None,
    *,
    dedup_enabled: bool,
    num_perm: int,
    shingle_size: int,
    resume_hashes=None,
) -> Iterator[FilteredDoc]:
    for doc in iter_documents(raw_root, files, progress=progress):
        raw_bytes = len(doc.text.encode("utf-8"))
        text = clean_text(doc.text, doc.source)
        keep, reason = should_keep_document(text, config, doc.source)
        if keep:
            exact_hash = compute_exact_hash(text)
            signature = None
            if dedup_enabled and not _hash_is_in_sorted_array(exact_hash, resume_hashes):
                signature = compute_minhash_signature(
                    text, num_perm=num_perm, shingle_size=shingle_size
                )
        else:
            exact_hash = None
            signature = None
        yield (
            doc.source,
            raw_bytes,
            keep,
            None if keep else reason,
            text if keep else None,
            exact_hash,
            signature,
        )


def _doc_stream_parallel(
    raw_root: Path,
    files: list[Path],
    config: SpakieConfig,
    progress: tqdm | None,
    workers: int,
    *,
    dedup_enabled: bool,
    num_perm: int,
    shingle_size: int,
    resume_hashes_path: str = "",
) -> Iterator[FilteredDoc]:
    file_paths = [str(path) for path in files]
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(
            config,
            str(raw_root),
            dedup_enabled,
            num_perm,
            shingle_size,
            resume_hashes_path,
        ),
    ) as pool:
        try:
            # ``imap`` still processes files concurrently but buffers completed
            # later files until every earlier file has been yielded. Do not use
            # imap_unordered here: --resume reconstructs its cursor by replaying
            # this stream and subtracting the tokenized prefix.
            for result in pool.imap(_worker_process_file, file_paths, chunksize=1):
                if progress is not None:
                    progress.update(result["file_bytes"])
                for entry in result["documents"]:
                    yield entry
        except GeneratorExit:
            pool.terminate()
            raise


def resolve_worker_count(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return requested
    cores = os.cpu_count() or 1
    return max(1, cores // 2)


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts: dict[str, int] = defaultdict(int)
    for line in lines:
        counts[line] += 1
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / max(len(lines), 1)


def looks_boilerplate_heavy(text: str) -> bool:
    lower = text.lower()
    markers = (
        "privacy policy", "terms of service", "cookie policy", "all rights reserved",
        "sign in", "subscribe", "javascript required", "skip to content",
        "jump to navigation", "jump to search", "navigation menu",
        "click here to download", "click here to print",
    )
    return sum(marker in lower for marker in markers) >= 2


# Small, high-frequency English stopword set. Real prose hits several of these
# in any 200-word window; SEO spam, tag clouds, and link farms hit ~zero.
_CORE_STOPWORDS = frozenset({
    "the", "be", "to", "of", "and", "that", "have", "with",
    "is", "are", "was", "were", "in", "for", "on", "as",
})

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")


def _mean_word_length_from_words(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _stopword_hit_count_from_words(words: list[str]) -> int:
    return sum(1 for w in words if w in _CORE_STOPWORDS)


def _symbol_word_ratio_from_words(text: str, word_count: int) -> float:
    if word_count == 0:
        return 1.0
    hash_count = text.count("#")
    ellipsis_count = text.count("...") + text.count("…")
    return (hash_count + ellipsis_count) / word_count


# Characters that don't count as "noise" alongside alphanumerics and whitespace.
_NOISE_ALLOWED_PUNCT = frozenset(".,;:!?-_'\"()[]{}")


def noise_and_top_char_share(text: str) -> tuple[float, float]:
    """Compute ``noise_ratio`` and ``top_char_share`` in a single pass.

    Both metrics otherwise walk every character in Python (``isalnum`` /
    ``isspace`` per char), which dominated the quality filter. ``Counter`` does
    the tally at C speed, then predicates run once per *distinct* character
    (typically <200) instead of once per occurrence.
    """
    if not text:
        return 1.0, 0.0
    counts = Counter(text)
    noisy = 0
    total_nonspace = 0
    max_nonspace = 0
    for ch, count in counts.items():
        is_space = ch.isspace()
        if is_space:
            continue
        total_nonspace += count
        if count > max_nonspace:
            max_nonspace = count
        if not (ch.isalnum() or ch in _NOISE_ALLOWED_PUNCT):
            noisy += count
    noise = noisy / len(text)
    top_share = max_nonspace / total_nonspace if total_nonspace else 0.0
    return noise, top_share


def _word_ngram_counts(words: list[str], n: int) -> Counter:
    if n <= 0 or len(words) < n:
        return Counter()
    # Counter over tuples is materially faster than building joined strings.
    return Counter(zip(*(words[i:] for i in range(n))))


# Below ~50 words, n-gram char-share metrics are degenerate (the single longest
# n-gram dominates a tiny denominator). Skip rather than falsely reject.
_MIN_WORDS_FOR_NGRAM_METRIC = 50


def _ngram_str_len(ngram_tuple: tuple[str, ...]) -> int:
    # Length of the n-gram if it were joined with single spaces.
    return sum(len(w) for w in ngram_tuple) + max(0, len(ngram_tuple) - 1)


def _top_ngram_char_share_from_words(words: list[str], n: int, text_len: int) -> float:
    if len(words) < _MIN_WORDS_FOR_NGRAM_METRIC:
        return 0.0
    counts = _word_ngram_counts(words, n)
    if not counts:
        return 0.0
    top_ngram, top_count = max(counts.items(), key=lambda kv: kv[1])
    return (_ngram_str_len(top_ngram) * top_count) / max(text_len, 1)


def _dup_ngram_char_share_from_words(words: list[str], n: int, text_len: int) -> float:
    if len(words) < _MIN_WORDS_FOR_NGRAM_METRIC:
        return 0.0
    counts = _word_ngram_counts(words, n)
    dup_chars = sum(
        _ngram_str_len(ngram) * count for ngram, count in counts.items() if count > 1
    )
    return dup_chars / max(text_len, 1)


def url_email_line_ratio(text: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    flagged = sum(1 for ln in lines if _URL_RE.search(ln) or _EMAIL_RE.search(ln))
    return flagged / len(lines)


def pick_token_dtype(tokenizer: SpakieTokenizer):
    return np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32


def min_doc_chars_for_source(config: SpakieConfig, source: str) -> int:
    return config.source_min_doc_chars.get(source, config.min_doc_chars)


def recommended_tokenizer_threads(cpu_count: int | None = None) -> int:
    cores = cpu_count or os.cpu_count() or 1
    if cores <= 4:
        return max(1, cores)
    return max(1, min(MAX_RECOMMENDED_TOKENIZER_THREADS, cores - 2))


def token_shard_index(path: Path) -> int:
    match = re.fullmatch(r"tokens-(\d+)\.npy", path.name)
    if not match:
        raise ValueError(f"Unexpected token shard name: {path}")
    return int(match.group(1))


def list_token_shards(shard_dir: Path) -> list[Path]:
    paths = sorted(shard_dir.glob("tokens-*.npy"), key=token_shard_index)
    indices = [token_shard_index(path) for path in paths]
    expected = list(range(len(paths)))
    if indices != expected:
        raise ValueError(
            f"Token shards in {shard_dir} are not contiguous from tokens-00000.npy"
        )
    return paths


def count_shard_tokens(shard_paths: list[Path]) -> int:
    return sum(int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths)


def raw_input_contract(raw_root: Path, files: list[Path]) -> dict:
    entries = []
    total_bytes = 0
    for path in files:
        stat = path.stat()
        total_bytes += stat.st_size
        try:
            name = str(path.relative_to(raw_root))
        except ValueError:
            name = str(path)
        entries.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return {
        "schema_version": 1,
        "files": len(entries),
        "bytes": total_bytes,
        "fingerprint": stable_payload_sha256(entries),
    }


def preparation_contract(
    config: SpakieConfig,
    *,
    target_tokens: int,
    dedup: bool,
    source_glob: str | None,
    source_dirs: list[str] | None,
) -> dict:
    fields = (
        "min_doc_chars", "source_min_doc_chars", "max_repeated_line_ratio",
        "max_noise_ratio", "mean_word_length_min", "mean_word_length_max",
        "min_stopword_count", "max_symbol_word_ratio", "max_top_2gram_char_share",
        "max_top_3gram_char_share", "max_dup_5gram_char_share", "max_top_char_share",
        "max_url_email_line_ratio", "filter_profiles",
        "minimum_source_completion_ratio", "maximum_source_mix_deviation",
        "near_dup_jaccard_threshold", "near_dup_num_perm",
        "near_dup_shingle_size", "train_split_fraction", "token_shard_size",
    )
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "target_tokens": int(target_tokens),
        "dedup": bool(dedup),
        "source_glob": source_glob or "",
        "source_dirs": list(source_dirs or []),
        "config": {name: getattr(config, name) for name in fields},
        "source_plan": config.scaled_corpus_source_plan(
            target_processed_tokens=target_tokens
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class TokenShardWriter:
    def __init__(
        self,
        shard_dir: Path,
        shard_size: int,
        dtype,
        *,
        start_index: int = 0,
        existing_paths: list[Path] | None = None,
        vocab_size: int | None = None,
        max_token_id: int = -1,
    ):
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.dtype = dtype
        self.buffer = np.empty(shard_size, dtype=dtype)
        self.offset = 0
        self.index = start_index
        self.paths: list[Path] = list(existing_paths or [])
        self.vocab_size = vocab_size
        self.max_token_id = int(max_token_id)

    def add(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        min_id = min(token_ids)
        max_id = max(token_ids)
        if min_id < 0:
            raise ValueError(f"Tokenizer produced negative token ID {min_id}")
        if self.vocab_size is not None and max_id >= self.vocab_size:
            raise ValueError(
                f"Tokenizer produced ID {max_id} outside vocabulary size {self.vocab_size}"
            )
        self.max_token_id = max(self.max_token_id, max_id)
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
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing token shard: {path}")
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


def merge_shards(
    shard_paths: list[Path],
    train_path: Path,
    val_path: Path,
    train_fraction: float,
    dtype,
    *,
    train_tokens_target: int | None = None,
    show_progress: bool = False,
    source_runs: list[tuple[str, int, int]] | None = None,
    source_document_ends: dict[str, list[int] | array] | None = None,
    tokenizer_provenance: dict | None = None,
    preparation_provenance: dict | None = None,
    raw_input_provenance: dict | None = None,
    max_token_id: int | None = None,
) -> tuple[int, int]:
    total_tokens = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths)
    split_idx = int(total_tokens * train_fraction)
    source_train_targets: dict[str, int] | None = None
    if source_runs:
        has_gap = any(
            start != prev_end
            for (_, _, prev_end), (_, start, _) in zip(source_runs, source_runs[1:])
        )
        if source_runs[0][1] != 0 or has_gap:
            raise ValueError("source_runs must cover the flattened token stream without gaps")
        if source_runs[-1][2] != total_tokens:
            raise ValueError(
                f"source_runs end at {source_runs[-1][2]:,}, but shards contain {total_tokens:,} tokens"
            )
        source_totals: dict[str, int] = defaultdict(int)
        for source, start, end in source_runs:
            if end < start:
                raise ValueError(f"invalid source token range for {source}: {start}:{end}")
            source_totals[source] += end - start
        if train_tokens_target and 0 < train_tokens_target <= total_tokens:
            # Allocate an exact global target proportionally, using largest
            # remainders so every source contributes to the training split.
            raw_targets = {
                source: train_tokens_target * count / total_tokens
                for source, count in source_totals.items()
            }
            source_train_targets = {
                source: int(value) for source, value in raw_targets.items()
            }
            remainder = train_tokens_target - sum(source_train_targets.values())
            for source, _ in sorted(
                (
                    (source, raw_targets[source] - source_train_targets[source])
                    for source in source_totals
                ),
                key=lambda item: (-item[1], item[0]),
            )[:remainder]:
                source_train_targets[source] += 1
            split_idx = train_tokens_target
        else:
            source_train_targets = {
                source: int(count * train_fraction)
                for source, count in source_totals.items()
            }
            split_idx = sum(source_train_targets.values())
            if train_tokens_target and train_tokens_target > total_tokens:
                print(
                    f"WARNING: requested {train_tokens_target:,} train tokens but only "
                    f"{total_tokens:,} processed tokens are available. Falling back to "
                    f"per-source train_fraction={train_fraction:.4f} -> "
                    f"{split_idx:,} train / {total_tokens - split_idx:,} val tokens."
                )

        if source_document_ends:
            adjusted_targets: dict[str, int] = {}
            for source, target in source_train_targets.items():
                total = source_totals[source]
                boundaries = [int(value) for value in source_document_ends.get(source, [])]
                # Zero and the final document end are valid choices. A source
                # containing one document must go wholly to one split rather
                # than leaking a prefix into train and its suffix into val.
                candidates = [0]
                candidates.extend(value for value in boundaries if 0 < value < total)
                candidates.append(total)
                adjusted_targets[source] = min(candidates, key=lambda value: abs(value - target))
            source_train_targets = adjusted_targets
            split_idx = sum(source_train_targets.values())

    if train_tokens_target and train_tokens_target > split_idx and source_train_targets is None:
        if train_tokens_target > total_tokens:
            # Don't throw away a long, expensive prepare run just because the
            # corpus came up short of the configured train target. Fall back
            # to the natural train_fraction split and warn loudly so the user
            # knows to expand the corpus (or lower target_train_tokens) before
            # the next run.
            print(
                f"WARNING: requested {train_tokens_target:,} train tokens but only "
                f"{total_tokens:,} processed tokens are available. Falling back to "
                f"train_fraction={train_fraction:.4f} -> "
                f"{split_idx:,} train / {total_tokens - split_idx:,} val tokens."
            )
        else:
            split_idx = train_tokens_target

    if total_tokens and (split_idx <= 0 or split_idx >= total_tokens):
        raise ValueError(
            "Cannot create non-empty document-disjoint train and validation splits. "
            "Add more independent documents or adjust the split/target."
        )

    processed_dir = train_path.parent
    if val_path.parent != processed_dir:
        raise ValueError("train and validation arrays must share one directory")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # The manifest is the commit marker. Remove it before creating a new
    # generation, but leave any previously complete arrays in place until both
    # replacements have been fully written and fsynced.
    invalidate_processed_data(processed_dir)

    temp_paths: list[Path] = []
    train_arr = None
    val_arr = None
    try:
        for final_path in (train_path, val_path):
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{final_path.name}.", suffix=".tmp", dir=processed_dir
            )
            os.close(fd)
            temp_paths.append(Path(temp_name))
        temp_train_path, temp_val_path = temp_paths

        train_arr = np.lib.format.open_memmap(
            temp_train_path, mode="w+", dtype=dtype, shape=(split_idx,)
        )
        val_arr = np.lib.format.open_memmap(
            temp_val_path,
            mode="w+",
            dtype=dtype,
            shape=(total_tokens - split_idx,),
        )

        cursor = 0
        train_cursor = 0
        val_cursor = 0

        shard_lengths = [int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths]
        shard_starts = [0]
        for shard_len in shard_lengths[:-1]:
            shard_starts.append(shard_starts[-1] + shard_len)

        def copy_flat_range(
            start: int, end: int, destination, destination_cursor: int
        ) -> int:
            """Copy a half-open range from flat shards into an output array."""
            if end <= start:
                return destination_cursor
            shard_idx = bisect.bisect_right(shard_starts, start) - 1
            position = start
            while position < end:
                shard_start = shard_starts[shard_idx]
                shard_end = shard_start + shard_lengths[shard_idx]
                take_end = min(end, shard_end)
                shard = np.load(shard_paths[shard_idx], mmap_mode="r")
                local_start = position - shard_start
                local_end = take_end - shard_start
                amount = local_end - local_start
                destination[destination_cursor:destination_cursor + amount] = shard[
                    local_start:local_end
                ]
                destination_cursor += amount
                position = take_end
                shard_idx += 1
            return destination_cursor

        with tqdm(
            total=total_tokens,
            desc="Merging shards",
            unit="tok",
            unit_scale=True,
            disable=not show_progress,
        ) as progress:
            if source_runs and source_train_targets is not None:
                source_seen: dict[str, int] = defaultdict(int)
                for source, run_start, run_end in source_runs:
                    run_len = run_end - run_start
                    source_offset = source_seen[source]
                    train_limit = source_train_targets[source]
                    train_amount = max(0, min(run_len, train_limit - source_offset))
                    train_cursor = copy_flat_range(
                        run_start, run_start + train_amount, train_arr, train_cursor
                    )
                    val_cursor = copy_flat_range(
                        run_start + train_amount, run_end, val_arr, val_cursor
                    )
                    source_seen[source] += run_len
                    cursor = run_end
                    progress.update(run_len)
            else:
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
                    progress.update(shard_len)

        expected_val_tokens = total_tokens - split_idx
        if cursor != total_tokens or train_cursor != split_idx or val_cursor != expected_val_tokens:
            raise RuntimeError(
                "Shard merge cursor mismatch: "
                f"source={cursor}/{total_tokens}, train={train_cursor}/{split_idx}, "
                f"val={val_cursor}/{expected_val_tokens}"
            )

        train_arr.flush()
        val_arr.flush()
        del train_arr
        del val_arr
        train_arr = None
        val_arr = None
        for temp_path in temp_paths:
            with temp_path.open("rb") as handle:
                os.fsync(handle.fileno())

        os.replace(temp_train_path, train_path)
        temp_paths.remove(temp_train_path)
        os.replace(temp_val_path, val_path)
        temp_paths.remove(temp_val_path)
        publish_processed_data_manifest(
            train_path,
            val_path,
            train_tokens=split_idx,
            val_tokens=expected_val_tokens,
            dtype=dtype,
            tokenizer=tokenizer_provenance,
            preparation=preparation_provenance,
            raw_inputs=raw_input_provenance,
            max_token_id=max_token_id,
        )
        return split_idx, expected_val_tokens
    except BaseException:
        # Never leave a completion marker behind for a partially published
        # generation, including KeyboardInterrupt/SystemExit.
        invalidate_processed_data(processed_dir)
        raise
    finally:
        if train_arr is not None:
            del train_arr
        if val_arr is not None:
            del val_arr
        for temp_path in temp_paths:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def should_keep_document(text: str, config: SpakieConfig, source: str) -> tuple[bool, str]:
    if len(text) < min_doc_chars_for_source(config, source):
        return False, "too_short"

    # Compute the lowercased word list exactly once and reuse it across every
    # downstream feature. Previously each helper re-lowered the text and
    # re-ran `_SHINGLE_WORD_RE.findall`, which dominated the filter cost on
    # long docs (~6x redundant scans per kept doc).
    lowered = text.lower()
    words_lower = _SHINGLE_WORD_RE.findall(lowered)
    word_count = len(words_lower)
    text_len = len(text)
    profile = config.filter_profile_for_source(source)
    source_kind = config.corpus_source_kind(source)

    mwl = _mean_word_length_from_words(words_lower)
    if bool(profile.get("apply_word_length", True)) and (
        mwl < config.mean_word_length_min or mwl > config.mean_word_length_max
    ):
        return False, "bad_word_length"
    if bool(profile.get("apply_stopwords", True)) and (
        _stopword_hit_count_from_words(words_lower[:200]) < config.min_stopword_count
    ):
        return False, "low_stopwords"
    if _symbol_word_ratio_from_words(text, word_count) > float(
        profile.get("max_symbol_word_ratio", config.max_symbol_word_ratio)
    ):
        return False, "symbol_heavy"
    # Single Counter pass yields both the noise ratio (checked now) and the
    # top-char share (checked below) without walking every character twice.
    noise, top_share = noise_and_top_char_share(text)
    if noise > float(profile.get("max_noise_ratio", config.max_noise_ratio)):
        return False, "too_noisy"
    if repeated_line_ratio(text) > float(
        profile.get("max_repeated_line_ratio", config.max_repeated_line_ratio)
    ):
        return False, "repeated_lines"
    if url_email_line_ratio(text) > float(
        profile.get("max_url_email_line_ratio", config.max_url_email_line_ratio)
    ):
        return False, "link_farm"
    if _top_ngram_char_share_from_words(words_lower, 2, text_len) > float(
        profile.get("max_top_2gram_char_share", config.max_top_2gram_char_share)
    ):
        return False, "repetitive_2gram"
    if _top_ngram_char_share_from_words(words_lower, 3, text_len) > float(
        profile.get("max_top_3gram_char_share", config.max_top_3gram_char_share)
    ):
        return False, "repetitive_3gram"
    if _dup_ngram_char_share_from_words(words_lower, 5, text_len) > float(
        profile.get("max_dup_5gram_char_share", config.max_dup_5gram_char_share)
    ):
        return False, "duplicate_5gram"
    if top_share > float(profile.get("max_top_char_share", config.max_top_char_share)):
        return False, "char_repetition"
    if source_kind != "code" and looks_boilerplate_heavy(text):
        return False, "boilerplate"
    return True, "kept"


def language_filter_sample(
    text: str, config: SpakieConfig, source: str
) -> str | None:
    """Return text suitable for language ID, or None when it must be skipped.

    Whole-file prose classification is actively harmful for source code and
    symbol-heavy mathematics. Code is language-neutral here; math is classified
    only from its natural-language words, never from formulas themselves.
    """
    profile = config.filter_profile_for_source(source)
    if not bool(profile.get("apply_language_id", True)):
        return None
    return language_id_sample(text, config.corpus_source_kind(source))


class TokenizerPipeline:
    """Background-thread SentencePiece encoder.

    Decouples the producer loop (worker IPC + dedup) from the cost of batched
    tokenization. The producer submits filled batches; a single daemon thread
    runs ``tokenizer.encode_batch`` (which releases the GIL) and pushes the
    encoded ids back. The producer drains finished batches opportunistically
    between submissions, so worker IPC stays saturated while encoding runs in
    parallel on native SentencePiece threads.
    """

    _SENTINEL = object()

    def __init__(self, tokenizer, num_threads: int, *, queue_depth: int = 2):
        self.tokenizer = tokenizer
        self.num_threads = num_threads
        self._in_q: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._out_q: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="tokenizer-pipeline", daemon=True
        )
        self._thread.start()

    def submit(self, batch: list[PendingTokenization]) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("TokenizerPipeline.submit after close_input")
        self._in_q.put(batch)

    def get_ready_blocking(
        self,
    ) -> tuple[list[PendingTokenization], list[list[int]]] | None:
        item = self._out_q.get()
        if item is self._SENTINEL:
            if self._error is not None:
                raise self._error
            return None
        return item

    def close_input(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._in_q.put(self._SENTINEL)

    def _run(self) -> None:
        try:
            while True:
                batch = self._in_q.get()
                if batch is self._SENTINEL:
                    self._out_q.put(self._SENTINEL)
                    return
                if not batch:
                    continue
                texts = [item.text for item in batch]
                if self.num_threads == 1 or len(texts) == 1:
                    eos_id = self.tokenizer.eos_id
                    encoded = [self.tokenizer.encode(t) + [eos_id] for t in texts]
                else:
                    encoded = self.tokenizer.encode_batch(
                        texts, add_eos=True, num_threads=self.num_threads
                    )
                self._out_q.put((batch, encoded))
        except BaseException as exc:
            self._error = exc
            self._out_q.put(self._SENTINEL)


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


def corpus_quality_gate(report: dict, config: SpakieConfig) -> dict:
    """Evaluate hard corpus-level gates and advisory per-source targets.

    Downloader token counts are estimates made before canonical filtering and
    cross-source deduplication. Requiring every source to retain 95% of that
    estimate made healthy runs fail after all expensive work was complete.
    Broad kind coverage is the quality invariant; exact source quotas remain
    visible as warnings so genuine shortfalls are still actionable.
    """
    failures: list[str] = []
    warnings: list[str] = []
    target_total = max(int(report.get("target_processed_tokens", 0)), 1)
    actual_total = max(int(report.get("processed_tokens", 0)), 0)
    source_targets = report.get("source_targets", {})
    source_stats = report.get("source_stats", {})

    corpus_completion = actual_total / target_total
    if corpus_completion < config.minimum_corpus_completion_ratio:
        failures.append(
            f"corpus: {corpus_completion:.1%} of processed-token target "
            f"({actual_total:,}/{target_total:,})"
        )

    target_by_kind: dict[str, int] = defaultdict(int)
    actual_by_kind: dict[str, int] = defaultdict(int)
    planned_sources: set[str] = set()
    for source, target in source_targets.items():
        target_tokens = int(target.get("target_tokens", 0))
        if not target.get("enabled", True) or target_tokens <= 0:
            continue
        planned_sources.add(source)
        kind = str(target.get("kind", "unknown"))
        target_by_kind[kind] += target_tokens
        actual_tokens = int(source_stats.get(source, {}).get("tokens_kept", 0))
        actual_by_kind[kind] += actual_tokens

        completion = actual_tokens / target_tokens
        if completion < config.minimum_source_completion_ratio:
            warnings.append(
                f"{source}: {completion:.1%} of token quota "
                f"({actual_tokens:,}/{target_tokens:,})"
            )
        target_share = target_tokens / target_total
        actual_share = actual_tokens / max(actual_total, 1)
        deviation = abs(actual_share - target_share)
        if deviation > config.maximum_source_mix_deviation:
            warnings.append(
                f"{source}: actual share {actual_share:.1%} differs from "
                f"target {target_share:.1%} by {deviation:.1%}"
            )

    unplanned_tokens = 0
    unplanned_sources: dict[str, int] = {}
    for source, stats in source_stats.items():
        if source in planned_sources:
            continue
        tokens = int(stats.get("tokens_kept", 0))
        if tokens <= 0:
            continue
        unplanned_sources[source] = tokens
        unplanned_tokens += tokens
        actual_by_kind[config.corpus_source_kind(source)] += tokens
    unplanned_share = unplanned_tokens / max(actual_total, 1)
    if unplanned_sources:
        detail = ", ".join(
            f"{source}={tokens:,}" for source, tokens in sorted(unplanned_sources.items())
        )
        warnings.append(f"unplanned sources contributed {unplanned_share:.1%}: {detail}")
    if unplanned_share > config.maximum_unplanned_source_share:
        failures.append(
            f"unplanned sources are {unplanned_share:.1%} of the corpus; "
            f"maximum is {config.maximum_unplanned_source_share:.1%}"
        )

    kind_stats: dict[str, dict[str, int | float]] = {}
    for kind, target_tokens in sorted(target_by_kind.items()):
        actual_tokens = actual_by_kind.get(kind, 0)
        completion = actual_tokens / target_tokens
        target_share = target_tokens / target_total
        actual_share = actual_tokens / max(actual_total, 1)
        deviation = abs(actual_share - target_share)
        kind_stats[kind] = {
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "completion_ratio": completion,
            "target_share": target_share,
            "actual_share": actual_share,
            "share_deviation": deviation,
        }
        if completion < config.minimum_kind_completion_ratio:
            failures.append(
                f"{kind} sources: {completion:.1%} of kind target "
                f"({actual_tokens:,}/{target_tokens:,})"
            )
        if deviation > config.maximum_kind_mix_deviation:
            failures.append(
                f"{kind} sources: actual share {actual_share:.1%} differs from "
                f"target {target_share:.1%} by {deviation:.1%}"
            )

    return {
        "minimum_corpus_completion_ratio": config.minimum_corpus_completion_ratio,
        "minimum_kind_completion_ratio": config.minimum_kind_completion_ratio,
        "maximum_kind_mix_deviation": config.maximum_kind_mix_deviation,
        "maximum_unplanned_source_share": config.maximum_unplanned_source_share,
        "minimum_source_completion_ratio_warning": config.minimum_source_completion_ratio,
        "maximum_source_mix_deviation_warning": config.maximum_source_mix_deviation,
        "corpus_completion_ratio": corpus_completion,
        "unplanned_source_share": unplanned_share,
        "kind_stats": kind_stats,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _source_layout_from_resume_journal(
    journal_path: Path,
    *,
    expected_tokens: int,
) -> tuple[list[tuple[str, int, int]], dict[str, array], dict[str, int], int]:
    """Rebuild split provenance without rereading or retokenizing raw text."""
    source_runs: list[tuple[str, int, int]] = []
    source_document_ends: dict[str, array] = defaultdict(lambda: array("Q"))
    source_tokens: dict[str, int] = defaultdict(int)
    total_tokens = 0
    records = 0
    pending_progress = 0
    with tqdm(
        total=expected_tokens,
        desc="Validating completed shards",
        unit="tok",
        unit_scale=True,
    ) as progress:
        for record in iter_accepted_documents(journal_path):
            records += 1
            token_start = total_tokens
            total_tokens += record.token_count
            pending_progress += record.token_count
            source_tokens[record.source] += record.token_count
            source_document_ends[record.source].append(source_tokens[record.source])
            if (
                source_runs
                and source_runs[-1][0] == record.source
                and source_runs[-1][2] == token_start
            ):
                source_runs[-1] = (
                    record.source,
                    source_runs[-1][1],
                    total_tokens,
                )
            else:
                source_runs.append((record.source, token_start, total_tokens))
            if records % 10_000 == 0:
                progress.update(pending_progress)
                pending_progress = 0
        progress.update(pending_progress)
    if total_tokens != expected_tokens:
        raise RuntimeError(
            "Accepted-document journal does not align with completed token shards "
            f"({total_tokens:,} journal tokens vs {expected_tokens:,} shard tokens)."
        )
    return source_runs, source_document_ends, dict(source_tokens), records


def try_fast_finalize_resume(
    *,
    config: SpakieConfig,
    report_dest: Path,
    shard_paths: list[Path],
    resume_journal: Path,
    resume_tokens: int,
    target_tokens: int,
    source_plan: dict[str, dict[str, int | str | bool]],
    discovered_files: int,
    discovered_raw_bytes: int,
    tokenizer_provenance: dict,
    preparation_provenance: dict,
    raw_input_provenance: dict,
    max_token_id: int,
    token_dtype,
    enforce_quality_gates: bool,
    full_corpus_run: bool,
) -> dict | None:
    """Publish a fully scanned shard run rejected only by the old final gate.

    Interrupted runs contain ``partial_token_shards`` and continue through the
    normal replay path. A completed run has a report whose exact token/source
    accounting can be checked against the durable journal before publication.
    """
    if not report_dest.exists() or not shard_paths or resume_tokens <= 0:
        return None
    try:
        report = json.loads(report_dest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        report.get("dry_run")
        or report.get("partial_token_shards")
        or int(report.get("processed_tokens", -1)) != resume_tokens
        or int(report.get("target_tokens_requested", -1)) != target_tokens
        or report.get("source_targets") != source_plan
        or int(report.get("discovered_files", -1)) != discovered_files
        or int(report.get("discovered_raw_bytes", -1)) != discovered_raw_bytes
    ):
        return None

    report["quality_gate"] = corpus_quality_gate(report, config)
    gate_failures = report["quality_gate"]["failures"]
    if enforce_quality_gates and full_corpus_run and gate_failures:
        formatted = "\n  - ".join(gate_failures)
        raise RuntimeError(
            "Completed token shards still fail corpus-level quality gates; "
            "processed arrays were not published:\n  - " + formatted
        )

    processed_dir = Path(config.processed_data_dir)
    already_valid, validation_detail = validate_processed_data(
        processed_dir,
        tokenizer_path=config.tokenizer_prefix + ".model",
        preparation=preparation_provenance,
        require_provenance=True,
    )
    if (
        already_valid
        and int(report.get("train_tokens", 0))
        + int(report.get("val_tokens", 0)) == resume_tokens
    ):
        print(f"Processed arrays are already published and valid: {validation_detail}")
        return report

    print(
        f"Found a completed {resume_tokens:,}-token shard run; "
        "skipping raw-corpus replay."
    )
    source_runs, source_document_ends, journal_source_tokens, record_count = (
        _source_layout_from_resume_journal(
            resume_journal,
            expected_tokens=resume_tokens,
        )
    )
    report_source_tokens = {
        source: int(stats.get("tokens_kept", 0))
        for source, stats in report.get("source_stats", {}).items()
        if int(stats.get("tokens_kept", 0)) > 0
    }
    if journal_source_tokens != report_source_tokens:
        raise RuntimeError(
            "Completed corpus report does not match per-source resume-journal totals."
        )
    if np.dtype(token_dtype) != np.dtype(np.uint16):
        raise ValueError("The current training stack expects uint16-compatible token ids")

    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    train_tokens, val_tokens = merge_shards(
        shard_paths,
        train_path,
        val_path,
        config.train_split_fraction,
        np.uint16,
        train_tokens_target=config.target_train_tokens,
        show_progress=True,
        source_runs=source_runs,
        source_document_ends=source_document_ends,
        tokenizer_provenance=tokenizer_provenance,
        preparation_provenance=preparation_provenance,
        raw_input_provenance=raw_input_provenance,
        max_token_id=max_token_id,
    )
    report.update({
        "processed_tokens": train_tokens + val_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "gap_to_target": max(target_tokens - train_tokens - val_tokens, 0),
        "resume": True,
        "resume_fast_finalize": True,
        "resume_existing_shards": len(shard_paths),
        "resume_existing_tokens": resume_tokens,
        "resume_journal_records": record_count,
        "scan_completed": True,
    })
    _write_json_atomic(report_dest, report)
    print(f"Train: {train_tokens:,} tokens -> {train_path}")
    print(f"Val:   {val_tokens:,} tokens -> {val_path}")
    print(f"Report: {report_dest}")
    return report


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
    resume: bool = False,
    tokenizer_threads: int | None = None,
    tokenize_batch_size: int = DEFAULT_TOKENIZE_BATCH_SIZE,
    tokenize_batch_chars: int = DEFAULT_TOKENIZE_BATCH_CHARS,
    workers: int | None = None,
    enforce_quality_gates: bool = True,
) -> dict:
    if resume and dry_run:
        raise ValueError("--resume cannot be combined with --dry_run")

    config = config or SpakieConfig()
    if target_train_tokens and target_train_tokens > 0:
        config.target_train_tokens = target_train_tokens
        config.pretrain_target_tokens = target_train_tokens
        config.refresh_derived_fields()
    target_tokens = target_tokens or config.target_processed_tokens
    source_plan = config.scaled_corpus_source_plan(target_processed_tokens=target_tokens)
    tokenizer = SpakieTokenizer(config.tokenizer_prefix + ".model")
    tokenizer_provenance = tokenizer_contract(config.tokenizer_prefix + ".model")
    if tokenizer.vocab_size != config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary mismatch: config expects {config.vocab_size:,} pieces, "
            f"but {config.tokenizer_prefix}.model contains {tokenizer.vocab_size:,}. "
            "Retrain tokenizer/train_tokenizer.py before rebuilding processed data."
        )
    token_dtype = pick_token_dtype(tokenizer)
    tokenizer_threads = tokenizer_threads or recommended_tokenizer_threads()
    tokenize_batch_size = max(1, tokenize_batch_size)
    tokenize_batch_chars = max(1, tokenize_batch_chars)
    worker_count = resolve_worker_count(workers)

    raw_root = Path(config.raw_data_dir).resolve()
    files = iter_input_files(raw_root, source_glob=source_glob, source_dirs=source_dirs)
    if not files:
        raise FileNotFoundError(f"No supported files found in {raw_root}")

    raw_bytes = sum(path.stat().st_size for path in files)
    raw_input_provenance = raw_input_contract(raw_root, files)
    preparation_provenance = preparation_contract(
        config,
        target_tokens=target_tokens,
        dedup=dedup,
        source_glob=source_glob,
        source_dirs=source_dirs,
    )
    run_contract = {
        "schema_version": 2,
        "tokenizer": tokenizer_provenance,
        "preparation": preparation_provenance,
        "raw_inputs": raw_input_provenance,
        "max_token_id": -1,
        "resume_journal": SHARD_RESUME_JOURNAL,
    }
    report_dest = Path(report_path or config.corpus_report_path)
    full_corpus_run = (
        target_tokens == config.target_processed_tokens
        and not source_glob
        and not source_dirs
    )
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
    dup_index: NearDuplicateIndex | None = None
    if dedup:
        dup_index = NearDuplicateIndex(
            threshold=config.near_dup_jaccard_threshold,
            num_perm=config.near_dup_num_perm,
            shingle_size=config.near_dup_shingle_size,
        )

    total_tokens = 0
    # Compressed provenance for the flattened token stream. A run is one
    # contiguous range emitted from the same corpus source; this lets the
    # final merge allocate train/validation tokens per source even when files
    # from different sources are interleaved.
    source_runs: list[tuple[str, int, int]] = []
    source_document_ends: dict[str, array] = defaultdict(lambda: array("Q"))
    shard_paths: list[Path] = []
    shard_dir = Path(config.token_shard_dir)
    shard_run_manifest = shard_dir / SHARD_RUN_MANIFEST
    resume_journal = shard_dir / SHARD_RESUME_JOURNAL
    resume_hashes_path = shard_dir / "resume_exact_hashes.npy"
    existing_shard_paths: list[Path] = []
    resume_tokens = 0
    if resume and not dry_run and shard_dir.exists():
        existing_shard_paths = list_token_shards(shard_dir)
        resume_tokens = count_shard_tokens(existing_shard_paths)
        if existing_shard_paths:
            if not shard_run_manifest.exists():
                raise RuntimeError(
                    f"Existing token shards have no provenance manifest {shard_run_manifest}; "
                    "rerun without --resume."
                )
            saved_run_contract = json.loads(
                shard_run_manifest.read_text(encoding="utf-8")
            )
            if saved_run_contract.get("schema_version") != run_contract["schema_version"]:
                raise RuntimeError(
                    "Existing token shards use an incompatible resume metadata schema; "
                    "rerun without --resume."
                )
            for key in ("tokenizer", "preparation", "raw_inputs"):
                if saved_run_contract.get(key) != run_contract[key]:
                    if key == "preparation":
                        saved_target = saved_run_contract.get(key, {}).get("target_tokens")
                        requested_target = run_contract[key].get("target_tokens")
                        if saved_target != requested_target:
                            raise RuntimeError(
                                "Cannot resume token preparation with a different target: "
                                f"existing shards target {saved_target:,} tokens, but this run "
                                f"requests {requested_target:,}. Rerun without --resume to start "
                                "a new shard set."
                            )
                    raise RuntimeError(
                        f"Existing token shards have a different {key} contract; "
                        "rerun without --resume."
                    )
            run_contract["max_token_id"] = int(
                saved_run_contract.get("max_token_id", -1)
            )
            if not resume_journal.exists():
                raise RuntimeError(
                    f"Existing token shards have no accepted-document journal "
                    f"{resume_journal}; rerun without --resume."
                )
            finalized_report = try_fast_finalize_resume(
                config=config,
                report_dest=report_dest,
                shard_paths=existing_shard_paths,
                resume_journal=resume_journal,
                resume_tokens=resume_tokens,
                target_tokens=target_tokens,
                source_plan=source_plan,
                discovered_files=len(files),
                discovered_raw_bytes=raw_bytes,
                tokenizer_provenance=tokenizer_provenance,
                preparation_provenance=preparation_provenance,
                raw_input_provenance=raw_input_provenance,
                max_token_id=int(run_contract.get("max_token_id", -1)),
                token_dtype=token_dtype,
                enforce_quality_gates=enforce_quality_gates,
                full_corpus_run=full_corpus_run,
            )
            if finalized_report is not None:
                return finalized_report
    if shard_dir.exists() and not dry_run and not resume:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run and not existing_shard_paths:
        _write_json_atomic(shard_run_manifest, run_contract)
        resume_journal.touch()
    start_shard_index = (
        token_shard_index(existing_shard_paths[-1]) + 1
        if existing_shard_paths
        else 0
    )
    writer = TokenShardWriter(
        shard_dir,
        shard_size=config.token_shard_size,
        dtype=token_dtype,
        start_index=start_shard_index,
        existing_paths=existing_shard_paths,
        vocab_size=tokenizer.vocab_size,
        max_token_id=int(run_contract.get("max_token_id", -1)),
    )
    resume_hash_values = array("Q")
    resume_record_count = 0
    resume_record_tokens = 0
    if resume_tokens:
        for record in iter_accepted_documents(resume_journal):
            resume_record_count += 1
            resume_record_tokens += record.token_count
            resume_hash_values.append(record.exact_hash)
            if dup_index is not None:
                dup_index.insert_known(record.exact_hash, record.band_keys)
        if resume_record_tokens != resume_tokens:
            raise RuntimeError(
                "Accepted-document resume journal does not align with existing token "
                f"shards ({resume_record_tokens:,} journal tokens vs "
                f"{resume_tokens:,} shard tokens); rerun without --resume."
            )
        sorted_hashes = np.frombuffer(resume_hash_values, dtype=np.uint64).copy()
        sorted_hashes.sort()
        np.save(resume_hashes_path, sorted_hashes)
        resume_hashes = np.load(resume_hashes_path, mmap_mode="r")
        resume_records = iter(iter_accepted_documents(resume_journal))
        expected_resume_record = next(resume_records, None)
    else:
        resume_hashes = None
        resume_records = iter(())
        expected_resume_record = None
    journal_handle = (
        resume_journal.open("ab", buffering=1024 * 1024) if not dry_run else None
    )
    pending: list[PendingTokenization] = []
    pending_chars = 0

    pipeline: TokenizerPipeline | None = None
    # Bound concurrency to 1 in-flight batch: while the encoder works on
    # batch N, the producer fills batch N+1. Draining is strictly FIFO and
    # always blocking, which keeps the accept/drop sequence deterministic
    # across runs (resume relies on byte-for-byte reproducibility) without
    # giving up the producer/encoder overlap.
    PIPELINE_DEPTH = 1
    in_flight = 0

    def process_encoded_batch(
        batch: list[PendingTokenization], encoded_batch: list[list[int]]
    ) -> bool:
        nonlocal total_tokens
        reached_target = False
        for item, token_ids in zip(batch, encoded_batch):
            stats = source_stats[item.source]
            source_cap = int(source_plan.get(item.source, {}).get("target_tokens", 0))
            # Dedup already ran in the producer loop, so we only need the
            # post-tokenization source-cap check here.
            if source_cap and stats["tokens_kept"] + len(token_ids) > source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue
            stats["documents_kept"] += 1
            stats["chars_kept"] += len(item.text)
            stats["tokens_kept"] += len(token_ids)
            source_document_ends[item.source].append(stats["tokens_kept"])
            token_start = total_tokens
            total_tokens += len(token_ids)
            token_end = total_tokens
            if (
                source_runs
                and source_runs[-1][0] == item.source
                and source_runs[-1][2] == token_start
            ):
                source_runs[-1] = (item.source, source_runs[-1][1], token_end)
            else:
                source_runs.append((item.source, token_start, token_end))
            writer.add(token_ids)
            if journal_handle is None or item.exact_hash is None:
                raise RuntimeError("accepted document is missing resume-journal state")
            band_keys = (
                dup_index.band_keys(item.signature)
                if dup_index is not None and item.signature is not None
                else ()
            )
            append_accepted_document(
                journal_handle,
                AcceptedDocument(
                    source=item.source,
                    token_count=len(token_ids),
                    char_count=len(item.text),
                    exact_hash=item.exact_hash,
                    band_keys=band_keys,
                ),
            )
            if total_tokens >= target_tokens:
                print(f"Reached target token budget: {total_tokens:,}")
                reached_target = True
                break
        return reached_target

    def submit_pending() -> bool:
        """Drain in-flight batches down to ``PIPELINE_DEPTH-1`` (blocking,
        FIFO), then submit ``pending``. Returns True if target was reached
        during the drain."""
        nonlocal pending, pending_chars, in_flight
        if not pending:
            return False
        if pipeline is None:
            raise RuntimeError("tokenizer pipeline is unavailable")
        while in_flight >= PIPELINE_DEPTH:
            ready = pipeline.get_ready_blocking()
            in_flight -= 1
            if ready is not None and process_encoded_batch(*ready):
                pending = []
                pending_chars = 0
                return True
        pipeline.submit(pending)
        in_flight += 1
        pending = []
        pending_chars = 0
        return False

    def drain_all() -> None:
        """Drain every remaining in-flight batch, blocking FIFO."""
        nonlocal in_flight
        if pipeline is None:
            raise RuntimeError("tokenizer pipeline is unavailable")
        while in_flight > 0:
            ready = pipeline.get_ready_blocking()
            in_flight -= 1
            if ready is None:
                continue
            if process_encoded_batch(*ready):
                # Target reached; drop the rest of the in-flight work.
                break
        # Eat any leftover results so the daemon thread isn't blocked on put.
        while in_flight > 0:
            pipeline.get_ready_blocking()
            in_flight -= 1

    if not dry_run:
        if resume:
            if resume_tokens:
                print(
                    f"Resuming from {len(existing_shard_paths):,} existing token shard(s) "
                    f"with {resume_tokens:,} token(s)"
                )
            else:
                print("Resume requested, but no existing token shards were found; starting fresh")
        print(
            "Tokenizing with SentencePiece requesting up to "
            f"{tokenizer_threads} worker thread(s) per batch, batches up to "
            f"{tokenize_batch_size:,} docs / {tokenize_batch_chars:,} chars"
        )
    if worker_count > 1:
        print(f"Filter workers: {worker_count} (per-file parallelism)")

    interrupted = False
    docs_progress = tqdm(
        total=raw_bytes,
        desc="Preparing data",
        unit="B",
        unit_scale=True,
        mininterval=0.5,
    )
    stream_kwargs = {
        "dedup_enabled": dedup,
        "num_perm": config.near_dup_num_perm,
        "shingle_size": config.near_dup_shingle_size,
    }
    if worker_count > 1:
        doc_stream = _doc_stream_parallel(
            raw_root,
            files,
            config,
            docs_progress,
            worker_count,
            resume_hashes_path=str(resume_hashes_path) if resume_tokens else "",
            **stream_kwargs,
        )
    else:
        doc_stream = _doc_stream_serial(
            raw_root,
            files,
            config,
            docs_progress,
            resume_hashes=resume_hashes,
            **stream_kwargs,
        )

    if not dry_run:
        pipeline = TokenizerPipeline(tokenizer, tokenizer_threads)
    try:
        for (
            source,
            raw_bytes_doc,
            kept,
            drop_reason,
            text,
            exact_hash,
            signature,
        ) in doc_stream:
            stats = source_stats[source]
            stats["documents_seen"] += 1
            stats["raw_bytes"] += raw_bytes_doc

            if not kept:
                stats["documents_dropped"] += 1
                stats["drop_reasons"][drop_reason] += 1
                continue

            if text is None:
                raise RuntimeError("accepted document has no cleaned text")
            # Language filtering belongs in canonical preparation, not only in
            # one optional downloader path: users can add raw files directly,
            # and old downloads may predate --english_only. Very short custom
            # documents are left alone because lid.176 is unreliable there;
            # production source minima are already at or above this threshold.
            language_sample = language_filter_sample(text, config, source)
            if language_sample is not None and not is_probably_english(
                language_sample, config
            ):
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["non_english"] += 1
                continue

            source_cap = int(source_plan.get(source, {}).get("target_tokens", 0))
            if source_cap and stats["tokens_kept"] >= source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue

            # A compatible resume journal lets us replay the already-written
            # prefix without rerunning SentencePiece or MinHash. Cleaning and
            # the exact content hash still verify deterministic input order.
            if expected_resume_record is not None:
                if exact_hash is None:
                    raise RuntimeError("resume candidate has no exact content hash")
                if (
                    source == expected_resume_record.source
                    and exact_hash == expected_resume_record.exact_hash
                ):
                    if len(text) != expected_resume_record.char_count:
                        raise RuntimeError(
                            "Resume journal content length differs from the cleaned raw document"
                        )
                    token_count = expected_resume_record.token_count
                    if source_cap and stats["tokens_kept"] + token_count > source_cap:
                        raise RuntimeError(
                            "Resume journal no longer fits the configured source token cap"
                        )
                    stats["documents_kept"] += 1
                    stats["chars_kept"] += len(text)
                    stats["tokens_kept"] += token_count
                    source_document_ends[source].append(stats["tokens_kept"])
                    token_start = total_tokens
                    total_tokens += token_count
                    if (
                        source_runs
                        and source_runs[-1][0] == source
                        and source_runs[-1][2] == token_start
                    ):
                        source_runs[-1] = (
                            source,
                            source_runs[-1][1],
                            total_tokens,
                        )
                    else:
                        source_runs.append((source, token_start, total_tokens))
                    expected_resume_record = next(resume_records, None)
                    if total_tokens >= target_tokens:
                        print(f"Reached target token budget: {total_tokens:,}")
                        break
                    continue

            # Dedup runs in the producer loop now (was previously post-encode)
            # so we don't burn tokenizer cycles on near-duplicates. Workers
            # already computed exact_hash + signature in parallel.
            if dup_index is not None:
                if exact_hash is None:
                    raise RuntimeError("deduplication candidate has no exact content hash")
                if dup_index.is_duplicate_exact(exact_hash):
                    stats["documents_dropped"] += 1
                    stats["drop_reasons"]["exact_duplicate"] += 1
                    continue
                if signature is None:
                    raise RuntimeError("deduplication candidate has no MinHash signature")
                if dup_index.query_signature(exact_hash, signature):
                    stats["documents_dropped"] += 1
                    stats["drop_reasons"]["near_duplicate"] += 1
                    continue

            if dry_run:
                estimated_tokens = max(1, int(len(text) / config.estimated_chars_per_token))
                stats["documents_kept"] += 1
                stats["chars_kept"] += len(text)
                if source_cap and stats["tokens_kept"] + estimated_tokens > source_cap:
                    stats["documents_dropped"] += 1
                    stats["drop_reasons"]["source_cap_reached"] += 1
                    continue
                stats["tokens_kept"] += estimated_tokens
                total_tokens += estimated_tokens
            else:
                pending.append(PendingTokenization(
                    source=source,
                    text=text,
                    exact_hash=exact_hash,
                    signature=signature,
                ))
                pending_chars += len(text)
                if len(pending) >= tokenize_batch_size or pending_chars >= tokenize_batch_chars:
                    if submit_pending():
                        break

            if stats["documents_seen"] % 25 == 0:
                docs_progress.set_postfix(
                    docs=f"{sum(s['documents_seen'] for s in source_stats.values()):,}",
                    kept=f"{sum(s['documents_kept'] for s in source_stats.values()):,}",
                    dropped=f"{sum(s['documents_dropped'] for s in source_stats.values()):,}",
                    tokens=f"{total_tokens:,}",
                    refresh=False,
                )

            if total_tokens >= target_tokens:
                print(f"Reached target token budget: {total_tokens:,}")
                break
    except KeyboardInterrupt:
        interrupted = True
        docs_progress.write(
            "\nInterrupted by Ctrl+C. Finalizing progress report and any completed token shards..."
        )
    finally:
        docs_progress.close()

    if expected_resume_record is not None and not interrupted:
        if pipeline is not None:
            pipeline.close_input()
        if journal_handle is not None:
            journal_handle.close()
        raise RuntimeError(
            "Existing token shards contain an accepted-document prefix that this run "
            "could not reproduce. Check raw inputs and prepare_data.py options."
        )

    if not dry_run:
        if pipeline is None:
            raise RuntimeError("tokenizer pipeline is unavailable")
        if pending and not interrupted:
            submit_pending()
        if not interrupted:
            drain_all()
        pipeline.close_input()
        shard_paths = writer.close()
        if journal_handle is None:
            raise RuntimeError("resume journal is unavailable during finalization")
        journal_handle.flush()
        os.fsync(journal_handle.fileno())
        journal_handle.close()
        run_contract["max_token_id"] = writer.max_token_id
        _write_json_atomic(shard_run_manifest, run_contract)

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
    report["resume"] = resume
    report["scan_completed"] = not interrupted
    report["quality_gate"] = corpus_quality_gate(report, config)
    gate_failures = report["quality_gate"]["failures"]
    if resume:
        report["resume_existing_shards"] = len(existing_shard_paths)
        report["resume_existing_tokens"] = resume_tokens
        report["resume_journal_records"] = resume_record_count
        report["resume_prefix_replayed"] = expected_resume_record is None

    processed_dir = Path(config.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dest.parent.mkdir(parents=True, exist_ok=True)
    with report_dest.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if interrupted:
        if shard_paths:
            report["partial_token_shards"] = [str(path) for path in shard_paths]
            with report_dest.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
        if not dry_run:
            # Existing arrays may still be useful for diagnosis or recovery,
            # but without a completion manifest the pipeline cannot mistake
            # them for the result of this interrupted invocation.
            invalidate_processed_data(processed_dir)
        print(f"Partial processed tokens: {total_tokens:,}")
        print(f"Partial report: {report_dest}")
        if shard_paths:
            print(f"Partial token shards: {shard_dir}")
        return report

    if dry_run:
        print(f"Estimated processed tokens: {total_tokens:,}")
        print(f"Gap to target: {max(target_tokens - total_tokens, 0):,}")
        return report

    if enforce_quality_gates and full_corpus_run and gate_failures:
        invalidate_processed_data(processed_dir)
        formatted = "\n  - ".join(gate_failures)
        raise RuntimeError(
            "Corpus quality gates failed; processed arrays were not published. "
            "Download/refill the missing source kinds or explicitly use "
            "--allow-incomplete-corpus for a diagnostic run:\n  - " + formatted
        )

    if not shard_paths:
        raise RuntimeError("No token shards were produced")

    output_dtype = np.uint16 if token_dtype == np.uint16 else np.uint32
    train_path = processed_dir / "train.npy"
    val_path = processed_dir / "val.npy"
    if output_dtype != np.uint16:
        raise ValueError("The current training stack expects uint16-compatible token ids")
    train_tokens, val_tokens = merge_shards(
        shard_paths,
        train_path,
        val_path,
        config.train_split_fraction,
        output_dtype,
        train_tokens_target=config.target_train_tokens,
        show_progress=True,
        source_runs=source_runs,
        source_document_ends=source_document_ends,
        tokenizer_provenance=tokenizer_provenance,
        preparation_provenance=preparation_provenance,
        raw_input_provenance=raw_input_provenance,
        max_token_id=writer.max_token_id,
    )

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
        "--resume",
        action="store_true",
        help="Continue from existing token shards instead of deleting and rebuilding them",
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Filter worker processes for clean+filter+MinHash-stream stage "
            "(0 = auto: half of CPU cores). Set to 1 to disable parallelism. "
            "Results are consumed in deterministic input-file order."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-corpus",
        action="store_true",
        help=(
            "Allow a full run to publish despite source quota/mix failures. "
            "Intended only for diagnostics and ablations."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dirs = [part.strip() for part in args.source_dirs.split(",") if part.strip()] or None
    try:
        prepare_data(
            target_tokens=args.target_tokens or None,
            target_train_tokens=args.target_train_tokens or None,
            dedup=args.dedup,
            report_path=args.report_path or None,
            source_glob=args.source_glob or None,
            source_dirs=source_dirs,
            dry_run=args.dry_run,
            resume=args.resume,
            tokenizer_threads=args.tokenizer_threads or None,
            tokenize_batch_size=args.tokenize_batch_size,
            tokenize_batch_chars=args.tokenize_batch_chars,
            workers=args.workers or None,
            enforce_quality_gates=not args.allow_incomplete_corpus,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Existing token shards were preserved for --resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
