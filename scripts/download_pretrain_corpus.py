"""Download a large pretraining corpus into resumable JSONL shards.

The downloader keeps the existing training contract intact by writing raw text
under data/raw/large_corpus/<source>/ and leaving tokenization to
scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import datasets
import requests
from datasets import load_dataset
from tqdm import tqdm

# The datasets library prints its own "Resolving data files" / download bars,
# which clash with our own tqdm bar under concurrent sources. We report our
# own progress, so silence theirs.
datasets.disable_progress_bars()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig, normalize_corpus_source
from runtime.langid import is_probably_english, load_langid_model


def _silence_resource_tracker_shutdown_race() -> None:
    """Suppress the cosmetic traceback from ``multiprocess``'s resource tracker.

    HuggingFace ``datasets`` uses the ``multiprocess`` package, whose
    ``ResourceTracker.__del__`` runs during interpreter shutdown and touches an
    ``RLock`` whose C internals may already be torn down, raising
    ``AttributeError: '_thread.RLock' object has no attribute '_recursion_count'``.
    The tracker's own server process still cleans up via pipe EOF, so the error
    is harmless — we just stop it from spamming the console on every exit.
    """
    try:
        from multiprocess import resource_tracker as _rt
    except Exception:
        return

    original_del = getattr(_rt.ResourceTracker, "__del__", None)
    if original_del is None:
        return

    def _safe_del(self, _orig=original_del):
        try:
            _orig(self)
        except Exception:
            # Interpreter shutdown races; the OS reaps the tracker regardless.
            pass

    _rt.ResourceTracker.__del__ = _safe_del


_silence_resource_tracker_shutdown_race()


def log(message: str) -> None:
    """Print without corrupting any active tqdm progress bars."""
    tqdm.write(message)


# Set when the user interrupts. Sources poll it via should_stop() so worker
# threads can finish the current record, flush progress, and exit cleanly.
STOP_EVENT = threading.Event()

HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}
GUTENDEX_API = "https://gutendex.com/books"
ARXIV_API = "https://export.arxiv.org/api/query"
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"
COMPACT_SEEN_HEX_LENGTH = 32
LEGACY_SEEN_IDS_MAX_BYTES = 256 * 1024 * 1024

HF_DATASETS: dict[str, dict] = {
    "fineweb-edu": {
        "variants": [
            {"path": "HuggingFaceFW/fineweb-edu", "name": None, "split": "train"},
            {"path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
        ],
        "text_fields": ("text", "content"),
        "id_fields": ("id", "doc_id", "url"),
        "url_fields": ("url", "source_url"),
        "title_fields": ("title",),
        "license": "ODC-By / research-friendly",
        "kind": "web",
    },
    "dolma": {
        "variants": [
            {"path": "allenai/dolma", "name": None, "split": "train"},
            {"path": "allenai/dolma", "name": "v1_7", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "doc_id"),
        "url_fields": ("url", "source"),
        "title_fields": ("title",),
        "license": "Research dataset",
        "kind": "web",
    },
    "refinedweb": {
        "variants": [
            {"path": "tiiuae/falcon-refinedweb", "name": None, "split": "train"},
        ],
        "text_fields": ("content", "text"),
        "id_fields": ("id", "url"),
        "url_fields": ("url",),
        "title_fields": ("title",),
        "license": "Research dataset",
        "kind": "web",
    },
    "fineweb_sample": {
        "variants": [
            {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "url"),
        "url_fields": ("url",),
        "title_fields": ("title",),
        "license": "ODC-By 1.0 / Common Crawl terms",
        "kind": "web",
    },
    "c4_en": {
        "variants": [
            {"path": "allenai/c4", "name": "en", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("url",),
        "url_fields": ("url",),
        "title_fields": ("title",),
        "license": "ODC-By 1.0 / Common Crawl terms",
        "kind": "web",
    },
    "wikipedia_snapshot": {
        "variants": [
            {"path": "wikimedia/wikipedia", "name": "20231101.en", "split": "train"},
            {"path": "wikipedia", "name": "20220301.en", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "title"),
        "url_fields": ("url",),
        "title_fields": ("title",),
        "license": "CC BY-SA 4.0",
        "kind": "reference",
    },
    "openwebmath": {
        "variants": [
            {"path": "open-web-math/open-web-math", "name": None, "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("url",),
        "url_fields": ("url",),
        "title_fields": ("title",),
        "license": "ODC-By 1.0 / Common Crawl terms",
        "kind": "technical",
    },
    "finemath": {
        "variants": [
            {"path": "HuggingFaceTB/finemath", "name": "finemath-4plus", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("url",),
        "url_fields": ("url",),
        "title_fields": (),
        "license": "ODC-By 1.0 / Common Crawl terms",
        "kind": "math",
    },
    "python_edu": {
        "variants": [
            {"path": "HuggingFaceTB/smollm-corpus", "name": "python-edu", "split": "train"},
        ],
        "text_fields": ("text", "content"),
        "id_fields": ("blob_id", "repo_name", "path"),
        "url_fields": (),
        "title_fields": ("path",),
        "license": "ODC-By; upstream repository licenses apply",
        "kind": "code",
    },
    "cosmopedia_v2": {
        "variants": [
            {"path": "HuggingFaceTB/smollm-corpus", "name": "cosmopedia-v2", "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("prompt",),
        "url_fields": (),
        # These fields are low-cardinality labels, not document titles. Using
        # them for title-level dedup rejects almost the entire stream.
        "title_fields": (),
        "license": "ODC-By",
        "kind": "synthetic_education",
    },
}

ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.DB",
    "cs.IR",
    "cs.LG",
    "cs.NE",
    "cs.RO",
    "cs.SE",
    "math.ST",
    "stat.ML",
]

STACKEXCHANGE_SITES = [
    "stackoverflow",
    "serverfault",
    "superuser",
    "askubuntu",
    "unix",
    "math",
    "stats",
    "datascience",
    "ai",
]


def _response_error_detail(response: requests.Response) -> str:
    """Pull the API's own error message out of the body, if there is one.

    requests.raise_for_status() only reports the HTTP status line, which is
    useless for e.g. Stack Exchange's JSON error bodies that explain *why*
    (quota exceeded, paging too deep, bad parameter, ...).
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(payload, dict):
        message = payload.get("error_message") or payload.get("message")
        if message:
            return str(message)
    return response.text[:200]


def api_get(url: str, *, params: dict | None = None, timeout: int = 60, max_retries: int = 5) -> requests.Response:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            # Network-level failure (timeout, connection reset, DNS, ...) is
            # genuinely transient — worth retrying with backoff.
            last_error = exc
            wait = min(2 ** attempt * 2, 30)
            log(f"  Request failed: {exc} ({wait}s)")
            time.sleep(wait)
            continue
        if response.status_code == 200:
            return response
        if response.status_code == 429 or response.status_code >= 500:
            wait = min(2 ** attempt * 3, 60)
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            log(f"  Retrying {url} after HTTP {response.status_code} ({wait}s)")
            time.sleep(wait)
            continue
        # Any other 4xx (bad request, not found, quota/throttle) is a
        # permanent rejection of this exact request — retrying it changes
        # nothing, so fail fast instead of burning a minute of backoff.
        detail = _response_error_detail(response)
        raise requests.HTTPError(f"{response.status_code} {response.reason} for {url} — {detail}", response=response)
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed request to {url}")


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = html.unescape(text)
    return normalize_text(text)


def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    return (name or "shard")[:max_len]


def pick_first(record: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def looks_navigation_heavy(text: str) -> bool:
    lower = text.lower()
    boilerplate_hits = sum(needle in lower for needle in (
        "privacy policy", "terms of service", "cookie policy", "all rights reserved",
        "sign in", "subscribe", "javascript", "enable cookies",
    ))
    return boilerplate_hits >= 2


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]

    start = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start = text.find("\n", idx)
            break

    end = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end = idx
            break
    return normalize_text(text[start:end])


def choose_text_url(formats: dict[str, str]) -> str | None:
    preferred = [
        "text/plain; charset=utf-8",
        "text/plain",
        "text/plain; charset=us-ascii",
    ]
    for key in preferred:
        value = formats.get(key)
        if value and not value.endswith(".zip"):
            return value
    for key, value in formats.items():
        if key.startswith("text/plain") and not value.endswith(".zip"):
            return value
    return None


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    items = []
    for entry in root.findall("atom:entry", ns):
        item_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns).strip()
        authors = [
            normalize_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)]
        if item_id and title and summary:
            items.append({
                "id": item_id,
                "title": title,
                "summary": summary,
                "published": published,
                "authors": authors,
                "categories": categories,
            })
    return items


def batched(values: list[int], size: int) -> list[list[int]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def fetch_answers(answer_ids: list[int], site: str) -> dict[int, dict]:
    answers = {}
    for batch in batched(answer_ids, 20):
        ids = ";".join(str(value) for value in batch)
        response = api_get(
            f"{STACKEXCHANGE_API}/answers/{ids}",
            params={"site": site, "filter": "withbody", "pagesize": len(batch)},
        )
        for item in response.json().get("items", []):
            answers[item["answer_id"]] = item
        time.sleep(0.5)
    return answers


@dataclass
class SourceBudget:
    source_name: str
    kind: str
    target_chars: int
    target_docs: int
    target_tokens_estimate: int


class JsonlShardWriter:
    def __init__(self, source_dir: Path, source_name: str, shard_char_limit: int, progress: dict):
        self.source_dir = source_dir
        self.source_name = source_name
        self.shard_char_limit = shard_char_limit
        self.progress = progress
        self.handle = None
        self.records: list[dict] = []
        self.current_chars = 0
        self.current_path: Path | None = None

    def _ensure_open(self) -> None:
        if self.handle is not None:
            return
        shard_index = int(self.progress.get("shard_index", 0))
        shard_name = f"shard-{shard_index:05d}"
        self.current_path = self.source_dir / f"{shard_name}.jsonl"
        self.handle = self.current_path.open("a", encoding="utf-8")
        self.current_chars = 0
        self.records = []

    def _finalize_current(self) -> None:
        if self.handle is None or self.current_path is None:
            return
        self.handle.close()
        manifest_path = self.current_path.with_suffix(".manifest.json")
        payload = {
            "source": self.source_name,
            "shard_path": self.current_path.name,
            "documents": len(self.records),
            "characters": self.current_chars,
            "records": self.records,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self.progress["shard_index"] = int(self.progress.get("shard_index", 0)) + 1
        self.handle = None
        self.records = []
        self.current_chars = 0
        self.current_path = None

    def write(self, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=False)
        estimated_chars = len(record["text"])
        if self.handle is not None and self.current_chars + estimated_chars > self.shard_char_limit and self.records:
            self._finalize_current()
        self._ensure_open()
        assert self.handle is not None
        self.handle.write(payload + "\n")
        self.current_chars += estimated_chars
        self.records.append({
            "id": record["id"],
            "title": record.get("title", ""),
            "chars": estimated_chars,
        })

    def close(self) -> None:
        self._finalize_current()


class SourceState:
    def __init__(
        self,
        source_dir: Path,
        budget: SourceBudget,
        resume: bool,
        config: SpakieConfig | None = None,
        progress_bar: tqdm | None = None,
        progress_lock: threading.Lock | None = None,
    ):
        self.source_dir = source_dir
        self.progress_path = source_dir / "progress.json"
        self.seen_ids_path = source_dir / "seen_ids.txt"
        self.seen_urls_path = source_dir / "seen_urls.txt"
        self.seen_titles_path = source_dir / "seen_titles.txt"
        self.budget = budget
        self.resume = resume
        self.config = config or SpakieConfig()
        self.progress = self._load_progress()
        self.writer = JsonlShardWriter(source_dir, budget.source_name, shard_char_limit=5_000_000, progress=self.progress)
        self.seen_ids = self._load_seen_ids()
        self.seen_urls = self._load_seen(self.seen_urls_path, namespace="url")
        self.seen_titles = self._load_seen(self.seen_titles_path, namespace="title")
        self.new_seen_ids: set[str] = set()
        self.new_seen_urls: set[str] = set()
        self.new_seen_titles: set[str] = set()
        self.last_save_time = 0.0
        # Shared across all concurrently running sources: one overall bar,
        # updated under a lock since tqdm's counter isn't safe for concurrent
        # writers from multiple threads.
        self.progress_bar = progress_bar
        self.progress_lock = progress_lock
        # HF streaming yields records in bursts (fast when a shard is already
        # buffered locally, stalled while the next one downloads). Updating
        # the bar on every doc makes tqdm's rate estimate swing wildly between
        # those bursts and stalls. Accumulate locally and flush on a steady
        # clock instead, so the shared bar sees evenly spaced updates.
        self.pending_bar_tokens = 0
        self.last_bar_flush_time = time.monotonic()

    @staticmethod
    def _seen_key(value: str, *, namespace: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if (
            len(normalized) == COMPACT_SEEN_HEX_LENGTH
            and all(char in "0123456789abcdef" for char in normalized)
        ):
            return normalized
        return hashlib.blake2b(
            f"{namespace}\0{normalized}".encode("utf-8"),
            digest_size=COMPACT_SEEN_HEX_LENGTH // 2,
        ).hexdigest()

    def _archive_oversized_legacy_seen_ids(self) -> bool:
        """Quarantine the old raw-prompt Cosmopedia index without reading it.

        Cosmopedia's prompt field can contain newlines, so the historical
        ``seen_ids.txt`` is not even a reliable one-record-per-line index. HF
        resume already has an exact row cursor; archive the legacy file and use
        compact hashes for newly accepted rows instead of materializing several
        gigabytes and tens of millions of fragments in a Python set.
        """
        path = self.seen_ids_path
        if (
            self.budget.source_name != "cosmopedia_v2"
            or not path.exists()
            or path.stat().st_size <= LEGACY_SEEN_IDS_MAX_BYTES
        ):
            return False
        if self.resume and int(self.progress.get("hf_rows_seen", 0)) <= 0:
            raise RuntimeError(
                "Cosmopedia has an oversized legacy seen_ids.txt but no HF row cursor. "
                "Run with --reset_source cosmopedia_v2 so resume cannot duplicate rows."
            )
        archive = path.with_name("seen_ids.legacy-raw.txt")
        suffix = 1
        while archive.exists():
            archive = path.with_name(f"seen_ids.legacy-raw.{suffix}.txt")
            suffix += 1
        os.replace(path, archive)
        log(
            f"  cosmopedia_v2: archived legacy raw seen-ID index "
            f"({archive.stat().st_size:,} bytes) as {archive.name}; "
            "HF row-cursor resume remains authoritative"
        )
        return True

    def _load_seen_ids(self) -> set[str]:
        if self._archive_oversized_legacy_seen_ids():
            return set()
        return self._load_seen(self.seen_ids_path, namespace="id")

    def _load_seen(self, path: Path, *, namespace: str) -> set[str]:
        values: set[str] = set()
        if self.resume and path.exists():
            # Stream and compact legacy values one at a time. Never build a set
            # of raw URLs/prompts/titles whose memory footprint dwarfs the file.
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    key = self._seen_key(line, namespace=namespace)
                    if key:
                        values.add(key)
        return values

    def _load_progress(self) -> dict:
        if self.resume and self.progress_path.exists():
            with self.progress_path.open("r", encoding="utf-8") as handle:
                progress = json.load(handle)
        else:
            progress = {
                "source": self.budget.source_name,
                "kind": self.budget.kind,
                "docs_written": 0,
                "chars_written": 0,
                "estimated_tokens": 0,
                "shard_index": 0,
                "hf_rows_seen": 0,
                "site_pages": {},
                "arxiv_offsets": {},
            }
        self._refresh_progress_metadata(progress)
        return progress

    def _refresh_progress_metadata(self, progress: dict) -> None:
        progress["source"] = self.budget.source_name
        progress["kind"] = self.budget.kind
        progress["target_chars"] = self.budget.target_chars
        progress["target_docs"] = self.budget.target_docs
        progress["target_tokens_estimate"] = self.budget.target_tokens_estimate
        progress["remaining_tokens_estimate"] = max(self.budget.target_tokens_estimate - int(progress.get("estimated_tokens", 0)), 0)
        progress["remaining_chars"] = max(self.budget.target_chars - int(progress.get("chars_written", 0)), 0)

    def save(self) -> None:
        self._flush_seen(self.seen_ids_path, self.new_seen_ids)
        self._flush_seen(self.seen_urls_path, self.new_seen_urls)
        self._flush_seen(self.seen_titles_path, self.new_seen_titles)
        self._refresh_progress_metadata(self.progress)
        with self.progress_path.open("w", encoding="utf-8") as handle:
            json.dump(self.progress, handle, ensure_ascii=False, indent=2)

    def _flush_seen(self, path: Path, values: set[str]) -> None:
        if not values:
            return
        with path.open("a", encoding="utf-8") as handle:
            for value in sorted(values):
                handle.write(value + "\n")
        values.clear()

    def maybe_save(self, min_interval: float = 10.0) -> None:
        """Persist progress at most once per interval to avoid disk churn."""
        now = time.monotonic()
        if now - self.last_save_time >= min_interval:
            self.save()
            self.last_save_time = now

    def maybe_flush_bar(self, min_interval: float = 0.5) -> None:
        """Push accumulated tokens to the shared bar on a steady clock.

        Flushing per-doc instead of on a timer makes tqdm's rate estimate
        swing wildly between bursty local batches and network-stalled ones;
        a steady flush interval smooths that out.
        """
        if self.progress_bar is None or self.progress_lock is None or self.pending_bar_tokens == 0:
            return
        now = time.monotonic()
        if now - self.last_bar_flush_time < min_interval:
            return
        with self.progress_lock:
            self.progress_bar.update(self.pending_bar_tokens)
        self.pending_bar_tokens = 0
        self.last_bar_flush_time = now

    def should_stop(self) -> bool:
        if STOP_EVENT.is_set():
            return True
        return (
            self.progress["chars_written"] >= self.budget.target_chars
            or (
                self.budget.target_docs > 0
                and self.progress["docs_written"] >= self.budget.target_docs
            )
            or self.progress["estimated_tokens"] >= self.budget.target_tokens_estimate
        )

    def accept(self, record: dict, english_only: bool) -> bool:
        text = normalize_text(record.get("text", ""))
        if len(text) < 400 or looks_navigation_heavy(text):
            return False
        if english_only and not is_probably_english(text, self.config):
            return False

        doc_id = str(record.get("id", "")).strip()
        title = record.get("title", "").strip().lower()
        url = record.get("url", "").strip().lower()
        doc_id_key = self._seen_key(doc_id, namespace="id")
        title_key = self._seen_key(title, namespace="title")
        url_key = self._seen_key(url, namespace="url")
        if doc_id_key and doc_id_key in self.seen_ids:
            return False
        if title_key and title_key in self.seen_titles:
            return False
        if url_key and url_key in self.seen_urls:
            return False

        record["text"] = text
        self.writer.write(record)
        token_delta = max(1, len(text) // 4)
        self.progress["docs_written"] += 1
        self.progress["chars_written"] += len(text)
        self.progress["estimated_tokens"] += token_delta
        self.pending_bar_tokens += token_delta
        self.maybe_flush_bar()
        if doc_id_key:
            self.seen_ids.add(doc_id_key)
            self.new_seen_ids.add(doc_id_key)
        if title_key:
            self.seen_titles.add(title_key)
            self.new_seen_titles.add(title_key)
        if url_key:
            self.seen_urls.add(url_key)
            self.new_seen_urls.add(url_key)
        return True

    def progress_summary(self) -> str:
        return (
            f"{self.progress['docs_written']:,} docs, "
            f"{self.progress['chars_written']:,} chars, "
            f"{self.progress['estimated_tokens']:,} est tokens, "
            f"{self.progress['remaining_tokens_estimate']:,} est tokens remaining"
        )

    def close(self) -> None:
        self.writer.close()
        self.save()
        if self.progress_bar is not None and self.progress_lock is not None and self.pending_bar_tokens:
            with self.progress_lock:
                self.progress_bar.update(self.pending_bar_tokens)
            self.pending_bar_tokens = 0


def build_budget(source_name: str, source_plan: dict[str, int | str | bool], max_docs: int) -> SourceBudget:
    target_chars = int(source_plan.get("target_raw_chars", 0))
    target_tokens = int(source_plan.get("target_tokens", 0))
    target_docs = max_docs if max_docs > 0 else 0
    return SourceBudget(
        source_name=source_name,
        kind=str(source_plan.get("kind", "unknown")),
        target_chars=target_chars,
        target_docs=target_docs,
        target_tokens_estimate=target_tokens,
    )


def reset_source_dir(source_dir: Path) -> None:
    if not source_dir.exists():
        return
    for path in source_dir.glob("*"):
        if path.is_file():
            path.unlink()


def iter_hf_records(source_name: str):
    spec = HF_DATASETS[source_name]
    last_error = None
    for variant in spec["variants"]:
        try:
            kwargs = {
                "path": variant["path"],
                "split": variant.get("split", "train"),
                "streaming": True,
            }
            if variant.get("name"):
                kwargs["name"] = variant["name"]
            dataset = load_dataset(**kwargs)
            for row in dataset:
                text = pick_first(row, spec["text_fields"])
                if not text:
                    continue
                title = pick_first(row, spec["title_fields"])
                url = pick_first(row, spec["url_fields"])
                if source_name == "wikipedia_snapshot" and title and not url:
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                yield {
                    "id": pick_first(row, spec["id_fields"]) or url or title or f"{source_name}-{random.getrandbits(64)}",
                    "title": title,
                    "url": url,
                    "text": text,
                    "meta": {
                        "source": source_name,
                        "dataset": variant["path"],
                        "config": variant.get("name"),
                        "license": spec["license"],
                    },
                }
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to stream dataset for {source_name}: {last_error}")


def ingest_hf_source(source_name: str, state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        log(f"  {source_name}: already at target, skipping download")
        return
    rows_seen = int(state.progress.get("hf_rows_seen", 0))
    record_iter = iter_hf_records(source_name)
    if rows_seen > 0:
        # HF streaming can't seek, so resume replays already-seen rows and skips
        # them here (cheap) rather than re-accepting them.
        record_iter = itertools.islice(record_iter, rows_seen, None)
    for record in record_iter:
        state.progress["hf_rows_seen"] += 1
        if state.should_stop():
            break
        state.accept(record, english_only=english_only)
        state.maybe_save()


def ingest_gutenberg(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        log("  gutenberg: already at target, skipping download")
        return
    consecutive_failures = 0
    next_url = GUTENDEX_API
    while next_url and not state.should_stop():
        payload = api_get(next_url).json()
        for book in payload.get("results", []):
            if state.should_stop():
                break
            if english_only and "en" not in book.get("languages", []):
                continue
            text_url = choose_text_url(book.get("formats", {}))
            if not text_url:
                continue
            try:
                # gutenberg.org throttles bulk fetchers by stalling connections,
                # so a generous timeout/retry budget here turns a blocked IP into
                # ~10 minutes of dead waiting per book. Keep the per-book cost low.
                raw_text = api_get(text_url, timeout=30, max_retries=2).text
            except Exception as exc:
                consecutive_failures += 1
                log(f"  Skip Gutenberg #{book['id']}: {exc}")
                if consecutive_failures >= 5:
                    # Like the Stack Exchange quota case: repeated timeouts mean
                    # the host is throttling this IP, and every remaining book
                    # would fail the same slow way. Stop and resume later.
                    log(
                        "  gutenberg: 5 consecutive fetch failures — gutenberg.org "
                        "is likely throttling this IP, stopping source early"
                    )
                    return
                continue
            consecutive_failures = 0
            text = strip_gutenberg_boilerplate(raw_text)
            authors = ", ".join(author["name"] for author in book.get("authors", [])) or "Unknown"
            state.accept({
                "id": str(book["id"]),
                "title": book["title"],
                "url": text_url,
                "text": f"# {book['title']}\n\nAuthors: {authors}\n\n{text}",
                "meta": {
                    "source": "gutenberg",
                    "book_id": book["id"],
                    "authors": authors,
                    "downloads": book.get("download_count", 0),
                    "license": "Public domain",
                },
            }, english_only=english_only)
            state.maybe_save()
            time.sleep(0.5)
        next_url = payload.get("next")


def ingest_arxiv(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        log("  arxiv: already at target, skipping download")
        return
    offsets = state.progress.setdefault("arxiv_offsets", {})
    while not state.should_stop():
        made_progress = False
        for category in ARXIV_CATEGORIES:
            if state.should_stop():
                break
            start = int(offsets.get(category, 0))
            params = {
                "search_query": f"cat:{category}",
                "start": start,
                "max_results": 200,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            items = parse_arxiv_feed(api_get(ARXIV_API, params=params).text)
            if not items:
                continue
            made_progress = True
            offsets[category] = start + len(items)
            for item in items:
                if state.should_stop():
                    break
                body = (
                    f"# {item['title']}\n\nPublished: {item['published']}\n"
                    f"Authors: {', '.join(item['authors'])}\n"
                    f"Categories: {', '.join(item['categories'])}\n\n"
                    f"## Abstract\n\n{item['summary']}"
                )
                state.accept({
                    "id": item["id"],
                    "title": item["title"],
                    "url": item["id"],
                    "text": body,
                    "meta": {"source": "arxiv", "license": "arXiv metadata / abstract"},
                }, english_only=english_only)
            state.maybe_save()
            time.sleep(0.5)
        if not made_progress:
            break


def _is_quota_exhausted(exc: Exception) -> bool:
    message = str(exc).lower()
    return "quota" in message or "throttle" in message or "too many requests" in message


def ingest_stackexchange(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        log("  stackexchange: already at target, skipping download")
        return
    site_pages = state.progress.setdefault("site_pages", {})
    for site in STACKEXCHANGE_SITES:
        page = int(site_pages.get(site, 1))
        while not state.should_stop():
            try:
                payload = api_get(
                    f"{STACKEXCHANGE_API}/questions",
                    params={
                        "site": site,
                        "page": page,
                        "pagesize": 100,
                        "order": "desc",
                        "sort": "votes",
                        "filter": "withbody",
                    },
                ).json()
                questions = payload.get("items", [])
                if not questions:
                    break
                accepted_ids = [q["accepted_answer_id"] for q in questions if q.get("accepted_answer_id")]
                answers = fetch_answers(accepted_ids, site)
            except Exception as exc:
                # Quota/throttle errors are IP-wide, not per-site — every
                # remaining site would fail the same way, so stop the whole
                # source instead of burning a doomed request per site.
                if _is_quota_exhausted(exc):
                    log(f"  stackexchange: quota/throttle exhausted, stopping source early: {exc}")
                    return
                log(f"  Skip Stack Exchange site '{site}': {exc}")
                break
            for question in questions:
                if state.should_stop():
                    break
                answer = answers.get(question.get("accepted_answer_id"))
                if not answer:
                    continue
                title = html.unescape(question["title"])
                question_text = html_to_text(question.get("body", ""))
                answer_text = html_to_text(answer.get("body", ""))
                text = (
                    f"# {title}\n\nSite: {site}\nTags: {', '.join(question.get('tags', []))}\n\n"
                    f"## Question\n\n{question_text}\n\n## Accepted Answer\n\n{answer_text}"
                )
                state.accept({
                    "id": f"{site}:{question['question_id']}",
                    "title": f"{site}::{title}",
                    "url": question.get("link", ""),
                    "text": text,
                    "meta": {
                        "source": "stackexchange",
                        "site": site,
                        "license": "CC BY-SA 4.0",
                    },
                }, english_only=english_only)
                time.sleep(0.2)
            page += 1
            site_pages[site] = page
            state.maybe_save()
            if not payload.get("has_more"):
                break
            time.sleep(0.5)


SOURCE_HANDLERS: dict[str, Callable[[SourceState, bool], None]] = {
    "gutenberg": ingest_gutenberg,
    "stackexchange": ingest_stackexchange,
    "arxiv": ingest_arxiv,
}


def parse_sources(raw: str, config: SpakieConfig) -> list[str]:
    requested = [normalize_corpus_source(part) for part in raw.split(",") if part.strip()]
    if not requested or requested == ["all"]:
        requested = [
            source_name
            for source_name, plan in config.corpus_source_plan.items()
            if plan.get("enabled", True)
        ]
    unique_sources: list[str] = []
    for source_name in requested:
        if source_name not in unique_sources:
            unique_sources.append(source_name)
    return unique_sources


def run_source(
    source_name: str,
    budget: SourceBudget,
    config: SpakieConfig,
    english_only: bool,
    resume: bool,
    progress_bar: tqdm,
    progress_lock: threading.Lock,
) -> None:
    """Download a single source end to end. Runs in its own worker thread."""
    source_dir = Path(config.large_corpus_dir) / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    state = SourceState(
        source_dir, budget, resume=resume, config=config,
        progress_bar=progress_bar, progress_lock=progress_lock,
    )
    try:
        if source_name in HF_DATASETS:
            ingest_hf_source(source_name, state, english_only=english_only)
        else:
            SOURCE_HANDLERS[source_name](state, english_only)
    finally:
        state.close()
    log(f"  completed [{source_name}]: {state.progress_summary()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a large pretraining corpus into JSONL shards")
    parser.add_argument("--sources", type=str, default="all", help="Comma-separated list of sources")
    parser.add_argument("--target_tokens_estimate", type=int, default=0, help="Estimated processed token target")
    parser.add_argument("--max_docs", type=int, default=0, help="Optional per-source document cap")
    parser.add_argument("--resume", action="store_true", help="Resume from progress manifests")
    parser.add_argument("--reset_source", type=str, default="", help="Comma-separated list of sources to reset")
    parser.add_argument(
        "--english-only", "--english_only", dest="english_only", action="store_true",
        help="Keep only likely English documents",
    )
    parser.add_argument("--workers", type=int, default=0, help="Concurrent source downloads (0 = one per source)")
    args = parser.parse_args()
    STOP_EVENT.clear()

    config = SpakieConfig()
    root = Path(config.large_corpus_dir)
    root.mkdir(parents=True, exist_ok=True)

    reset_sources = {normalize_corpus_source(part) for part in args.reset_source.split(",") if part.strip()}
    target_tokens = args.target_tokens_estimate or config.target_processed_tokens
    sources = parse_sources(args.sources, config)
    source_plan = config.scaled_corpus_source_plan(
        target_processed_tokens=target_tokens,
        requested_sources=sources,
    )

    budgets: dict[str, SourceBudget] = {}
    for source_name in sources:
        if source_name not in source_plan:
            raise ValueError(f"Unsupported source or disabled source plan entry: {source_name}")
        if source_name not in SOURCE_HANDLERS and source_name not in HF_DATASETS:
            raise ValueError(f"Unsupported source: {source_name}")
        source_dir = root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        if source_name in reset_sources:
            reset_source_dir(source_dir)
        budgets[source_name] = build_budget(source_name, source_plan[source_name], args.max_docs)

    # Load the langid model once up front so concurrent workers share it instead
    # of racing to download/load it on their first document.
    if args.english_only:
        load_langid_model(config, logger=log)

    workers = args.workers if args.workers > 0 else len(sources)
    workers = max(1, min(workers, len(sources)))
    log(f"Downloading {len(sources)} source(s) with {workers} concurrent worker(s)")

    total_target_tokens = sum(b.target_tokens_estimate for b in budgets.values())
    already_done = sum(
        int(json.loads((root / name / "progress.json").read_text()).get("estimated_tokens", 0))
        for name in sources
        if args.resume and (root / name / "progress.json").exists()
    )
    progress_bar = tqdm(
        total=total_target_tokens or None,
        initial=already_done,
        desc="Downloading corpus",
        unit="tok",
        unit_scale=True,
        # Lower smoothing = closer to a running average since start rather than
        # weighting recent (bursty) updates heavily, so the tok/s readout
        # doesn't swing wildly between buffered bursts and network stalls.
        smoothing=0.05,
    )
    progress_lock = threading.Lock()

    failures: list[tuple[str, str]] = []
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_source, name, budgets[name], config, args.english_only, args.resume,
                    progress_bar, progress_lock,
                ): name
                for name in sources
            }
            try:
                for future in as_completed(futures):
                    source_name = futures[future]
                    exc = future.exception()
                    if exc is not None:
                        failures.append((source_name, str(exc)))
                        log(f"  source failed: {source_name} -> {exc}")
            except KeyboardInterrupt:
                interrupted = True
                log("\nInterrupted; signalling workers to flush progress and stop...")
                STOP_EVENT.set()
                # Exiting the `with` block waits for workers to reach should_stop()
                # and close() their shard writers, so progress stays resumable.
    finally:
        progress_bar.close()

    if failures:
        log("\nCompleted with source failures:")
        for source_name, message in failures:
            log(f"  - {source_name}: {message}")
    if interrupted:
        return 130
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        log("\nInterrupted while downloading pretraining data.")
        sys.exit(130)
