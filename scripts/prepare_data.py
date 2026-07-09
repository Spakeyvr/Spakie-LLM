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
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import tempfile
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import xxhash
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig, normalize_corpus_source
from runtime.langid import is_probably_english
from runtime.processed_data import (
    invalidate_processed_data,
    publish_processed_data_manifest,
)
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
    # When dedup is enabled these are pre-computed by the worker (or serial
    # path) and only used by the main process for fast dedup lookups.
    exact_hash: int | None = None
    signature: np.ndarray | None = None


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

# Wikipedia/citation residue: [edit], [1], [42], [citation needed], [note 3], [nb 2].
# Constrained to short bracketed tokens so we don't strip URLs or date stamps.
_CITATION_RE = re.compile(
    r"\[(?:edit|citation needed|note\s+\d+|nb\s+\d+|\d{1,3})\]",
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


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [
        ln for ln in text.splitlines()
        if not _NAV_LINE_RE.match(ln) and not _JS_CSS_LINE_RE.match(ln)
    ]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SHINGLE_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _iter_shingles(text: str, n: int):
    words = _SHINGLE_WORD_RE.findall(text.lower())
    if not words:
        return
    if len(words) < n:
        yield " ".join(words)
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


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

    Fully vectorized: all shingle hashes are computed once, then the permuted
    min-hash reduction runs as blocked numpy matrix ops instead of a Python
    per-shingle ``MinHash.update`` loop (~3.5x faster on real docs). Output is
    byte-identical to the datasketch path. The returned uint64 array can be
    shipped across process boundaries and fed straight to the LSH index.
    """
    a, b, prime, max_hash = _minhash_params(num_perm)
    shingles = list(_iter_shingles(text, max(1, shingle_size)))
    if not shingles:
        return np.full(num_perm, max_hash, dtype=np.uint64)

    hv = np.fromiter(
        (xxhash.xxh32_intdigest(s.encode("utf-8")) for s in shingles),
        dtype=np.uint64,
        count=len(shingles),
    )
    running = np.full(num_perm, max_hash, dtype=np.uint64)
    for start in range(0, hv.shape[0], _MINHASH_BLOCK):
        chunk = hv[start:start + _MINHASH_BLOCK]
        # uint64 arithmetic wraps mod 2^64 elementwise, exactly matching
        # datasketch's scalar (a*hv + b) before the Mersenne modulo.
        phv = ((chunk[:, None] * a[None, :] + b[None, :]) % prime) & max_hash
        np.minimum(running, phv.min(axis=0), out=running)
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
        self.num_perm = num_perm
        self.shingle_size = max(1, shingle_size)
        self._ranges: list[tuple[int, int]] = [
            (i * rows, (i + 1) * rows) for i in range(bands)
        ]
        self._tables: list[set[bytes]] = [set() for _ in range(bands)]
        self._exact_hashes: set[int] = set()

    # Convenience wrapper for the serial path: hash + signature + LSH check.
    def is_duplicate(self, text: str) -> bool:
        exact_hash = compute_exact_hash(text)
        if exact_hash in self._exact_hashes:
            return True
        signature = compute_minhash_signature(
            text, num_perm=self.num_perm, shingle_size=self.shingle_size
        )
        return self._check_and_insert(exact_hash, signature)

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
        band_keys = [signature[start:end].tobytes() for start, end in self._ranges]
        for table, key in zip(self._tables, band_keys):
            if key in table:
                self._exact_hashes.add(exact_hash)
                return True
        for table, key in zip(self._tables, band_keys):
            table.add(key)
        self._exact_hashes.add(exact_hash)
        return False


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


def _worker_init(
    config: SpakieConfig,
    raw_root_str: str,
    dedup_enabled: bool,
    num_perm: int,
    shingle_size: int,
) -> None:
    global _WORKER_CONFIG, _WORKER_RAW_ROOT
    global _WORKER_DEDUP_ENABLED, _WORKER_NUM_PERM, _WORKER_SHINGLE_SIZE
    _WORKER_CONFIG = config
    _WORKER_RAW_ROOT = Path(raw_root_str)
    _WORKER_DEDUP_ENABLED = dedup_enabled
    _WORKER_NUM_PERM = num_perm
    _WORKER_SHINGLE_SIZE = shingle_size


def _worker_process_file(file_path_str: str) -> dict:
    assert _WORKER_CONFIG is not None and _WORKER_RAW_ROOT is not None
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
        text = clean_text(doc.text)
        keep, reason = should_keep_document(text, config, doc.source)
        if keep and dedup_enabled:
            exact_hash = compute_exact_hash(text)
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
) -> Iterator[FilteredDoc]:
    for doc in iter_documents(raw_root, files, progress=progress):
        raw_bytes = len(doc.text.encode("utf-8"))
        text = clean_text(doc.text)
        keep, reason = should_keep_document(text, config, doc.source)
        if keep and dedup_enabled:
            exact_hash = compute_exact_hash(text)
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
) -> Iterator[FilteredDoc]:
    file_paths = [str(path) for path in files]
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(config, str(raw_root), dedup_enabled, num_perm, shingle_size),
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


def mean_word_length(text: str) -> float:
    words = _SHINGLE_WORD_RE.findall(text)
    return _mean_word_length_from_words(words)


def _mean_word_length_from_words(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def stopword_hit_count(text: str, max_words: int = 200) -> int:
    words = _SHINGLE_WORD_RE.findall(text.lower())[:max_words]
    return _stopword_hit_count_from_words(words)


def _stopword_hit_count_from_words(words: list[str]) -> int:
    return sum(1 for w in words if w in _CORE_STOPWORDS)


def symbol_word_ratio(text: str) -> float:
    words = _SHINGLE_WORD_RE.findall(text)
    return _symbol_word_ratio_from_words(text, len(words))


def _symbol_word_ratio_from_words(text: str, word_count: int) -> float:
    if word_count == 0:
        return 1.0
    hash_count = text.count("#")
    ellipsis_count = text.count("...") + text.count("…")
    return (hash_count + ellipsis_count) / word_count


def top_char_share(text: str) -> float:
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for ch in text:
        if not ch.isspace():
            counts[ch] += 1
            total += 1
    if total == 0:
        return 0.0
    return max(counts.values()) / total


# Characters that don't count as "noise" alongside alphanumerics and whitespace.
_NOISE_ALLOWED_PUNCT = frozenset(".,;:!?-_'\"()[]{}")


def noise_and_top_char_share(text: str) -> tuple[float, float]:
    """Compute ``noise_ratio`` and ``top_char_share`` in a single pass.

    Both metrics otherwise walk every character in Python (``isalnum`` /
    ``isspace`` per char), which dominated the quality filter. ``Counter`` does
    the tally at C speed, then predicates run once per *distinct* character
    (typically <200) instead of once per occurrence. Results are identical to
    calling ``noise_ratio(text)`` and ``top_char_share(text)`` separately.
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


def top_ngram_char_share(text: str, n: int) -> float:
    """Gopher-style: characters in the single most common word n-gram, over total chars."""
    words = _SHINGLE_WORD_RE.findall(text.lower())
    return _top_ngram_char_share_from_words(words, n, len(text))


def _top_ngram_char_share_from_words(words: list[str], n: int, text_len: int) -> float:
    if len(words) < _MIN_WORDS_FOR_NGRAM_METRIC:
        return 0.0
    counts = _word_ngram_counts(words, n)
    if not counts:
        return 0.0
    top_ngram, top_count = max(counts.items(), key=lambda kv: kv[1])
    return (_ngram_str_len(top_ngram) * top_count) / max(text_len, 1)


def dup_ngram_char_share(text: str, n: int) -> float:
    """Gopher-style: characters covered by any word n-gram appearing more than once."""
    words = _SHINGLE_WORD_RE.findall(text.lower())
    return _dup_ngram_char_share_from_words(words, n, len(text))


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


class TokenShardWriter:
    def __init__(
        self,
        shard_dir: Path,
        shard_size: int,
        dtype,
        *,
        start_index: int = 0,
        existing_paths: list[Path] | None = None,
    ):
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.dtype = dtype
        self.buffer = np.empty(shard_size, dtype=dtype)
        self.offset = 0
        self.index = start_index
        self.paths: list[Path] = list(existing_paths or [])

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
) -> tuple[int, int]:
    total_tokens = sum(int(np.load(path, mmap_mode="r").shape[0]) for path in shard_paths)
    split_idx = int(total_tokens * train_fraction)
    if train_tokens_target and train_tokens_target > split_idx:
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
        with tqdm(
            total=total_tokens,
            desc="Merging shards",
            unit="tok",
            unit_scale=True,
            disable=not show_progress,
        ) as progress:
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

    mwl = _mean_word_length_from_words(words_lower)
    if mwl < config.mean_word_length_min or mwl > config.mean_word_length_max:
        return False, "bad_word_length"
    if _stopword_hit_count_from_words(words_lower[:200]) < config.min_stopword_count:
        return False, "low_stopwords"
    if _symbol_word_ratio_from_words(text, word_count) > config.max_symbol_word_ratio:
        return False, "symbol_heavy"
    # Single Counter pass yields both the noise ratio (checked now) and the
    # top-char share (checked below) without walking every character twice.
    noise, top_share = noise_and_top_char_share(text)
    if noise > config.max_noise_ratio:
        return False, "too_noisy"
    if repeated_line_ratio(text) > config.max_repeated_line_ratio:
        return False, "repeated_lines"
    if url_email_line_ratio(text) > config.max_url_email_line_ratio:
        return False, "link_farm"
    if _top_ngram_char_share_from_words(words_lower, 2, text_len) > config.max_top_2gram_char_share:
        return False, "repetitive_2gram"
    if _top_ngram_char_share_from_words(words_lower, 3, text_len) > config.max_top_3gram_char_share:
        return False, "repetitive_3gram"
    if _dup_ngram_char_share_from_words(words_lower, 5, text_len) > config.max_dup_5gram_char_share:
        return False, "duplicate_5gram"
    if top_share > config.max_top_char_share:
        return False, "char_repetition"
    if looks_boilerplate_heavy(text):
        return False, "boilerplate"
    return True, "kept"


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

    def try_get_ready(self) -> tuple[list[PendingTokenization], list[list[int]]] | None:
        try:
            item = self._out_q.get_nowait()
        except queue.Empty:
            return None
        if item is self._SENTINEL:
            return None
        return item

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
    shard_paths: list[Path] = []
    shard_dir = Path(config.token_shard_dir)
    existing_shard_paths: list[Path] = []
    resume_tokens = 0
    if resume and not dry_run and shard_dir.exists():
        existing_shard_paths = list_token_shards(shard_dir)
        resume_tokens = count_shard_tokens(existing_shard_paths)
    if shard_dir.exists() and not dry_run and not resume:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
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
    )
    pending: list[PendingTokenization] = []
    pending_chars = 0
    resume_tokens_remaining = resume_tokens

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
        nonlocal total_tokens, resume_tokens_remaining
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
            total_tokens += len(token_ids)
            if resume_tokens_remaining:
                if len(token_ids) > resume_tokens_remaining:
                    raise RuntimeError(
                        "Existing token shards end in the middle of an accepted document. "
                        "Remove the shard directory and rerun without --resume."
                    )
                resume_tokens_remaining -= len(token_ids)
            else:
                writer.add(token_ids)
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
        assert pipeline is not None
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
        assert pipeline is not None
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
            raw_root, files, config, docs_progress, worker_count, **stream_kwargs
        )
    else:
        doc_stream = _doc_stream_serial(
            raw_root, files, config, docs_progress, **stream_kwargs
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

            assert text is not None
            # Language filtering belongs in canonical preparation, not only in
            # one optional downloader path: users can add raw files directly,
            # and old downloads may predate --english_only. Very short custom
            # documents are left alone because lid.176 is unreliable there;
            # production source minima are already at or above this threshold.
            if len(text) >= 400 and not is_probably_english(text, config):
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["non_english"] += 1
                continue

            source_cap = int(source_plan.get(source, {}).get("target_tokens", 0))
            if source_cap and stats["tokens_kept"] >= source_cap:
                stats["documents_dropped"] += 1
                stats["drop_reasons"]["source_cap_reached"] += 1
                continue

            # Dedup runs in the producer loop now (was previously post-encode)
            # so we don't burn tokenizer cycles on near-duplicates. Workers
            # already computed exact_hash + signature in parallel.
            if dup_index is not None:
                assert exact_hash is not None and signature is not None
                if dup_index.is_duplicate_exact(exact_hash):
                    stats["documents_dropped"] += 1
                    stats["drop_reasons"]["exact_duplicate"] += 1
                    continue
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

    if resume_tokens_remaining and not interrupted and total_tokens < target_tokens:
        raise RuntimeError(
            "Existing token shards contain more accepted tokens than this run could reproduce. "
            "Check that the raw inputs and prepare_data.py options match the interrupted run."
        )

    if not dry_run:
        assert pipeline is not None
        if pending and not interrupted:
            submit_pending()
        if not interrupted:
            drain_all()
        pipeline.close_input()
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
    report["resume"] = resume
    if resume:
        report["resume_existing_shards"] = len(existing_shard_paths)
        report["resume_existing_tokens"] = resume_tokens
        report["resume_tokens_remaining"] = resume_tokens_remaining

    processed_dir = Path(config.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dest = Path(report_path or config.corpus_report_path)
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
        resume=args.resume,
        tokenizer_threads=args.tokenizer_threads or None,
        tokenize_batch_size=args.tokenize_batch_size,
        tokenize_batch_chars=args.tokenize_batch_chars,
        workers=args.workers or None,
    )


if __name__ == "__main__":
    main()
