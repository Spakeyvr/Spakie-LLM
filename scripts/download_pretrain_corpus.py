"""Download a large pretraining corpus into resumable JSONL shards.

The downloader keeps the existing training contract intact by writing raw text
under data/raw/large_corpus/<source>/ and leaving tokenization to
scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import html
import itertools
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import datasets
import requests
from datasets import load_dataset
from requests.adapters import HTTPAdapter
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
SOFTWARE_HERITAGE_CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content"
COMPACT_SEEN_HEX_LENGTH = 32
LEGACY_SEEN_IDS_MAX_BYTES = 256 * 1024 * 1024
LEGACY_CURSOR_DEFER_MIN_TOKENS = 10_000_000
LEGACY_CURSOR_DEFER_FRACTION = 0.005
CURRENT_RATE_WINDOW_SECONDS = 15.0
INTERRUPT_GRACE_SECONDS = 5.0
MIN_DOCUMENT_CHARS = 400
JSONL_BUFFER_BYTES = 1024 * 1024
PYTHON_EDU_FETCH_AHEAD = 4
DEFAULT_ITEM_WORKERS = min(32, max(8, (os.cpu_count() or 8) * 2))
DEFAULT_HF_WORKERS = 4


_HTTP_CLIENTS = threading.local()


def _http_session() -> requests.Session:
    """Return one keep-alive HTTP session per long-lived worker thread."""
    session = getattr(_HTTP_CLIENTS, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        # Each session belongs to one worker thread. A one-connection pool is
        # enough and, unlike requests.get(), reuses TLS connections across the
        # millions of small Python-Edu object requests.
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, pool_block=True)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _HTTP_CLIENTS.session = session
    return session


class AcceptedRateMonitor:
    """Display recent accepted-token throughput, including zero-rate stalls."""

    def __init__(self, progress_bar: tqdm, progress_lock: threading.Lock):
        self.progress_bar = progress_bar
        self.progress_lock = progress_lock
        self.stop_event = threading.Event()
        now = time.monotonic()
        self.samples: deque[tuple[float, int]] = deque([(now, int(progress_bar.n))])
        self.thread = threading.Thread(target=self._run, name="accepted-rate", daemon=True)
        self.thread.start()

    @staticmethod
    def _format_rate(rate: float) -> str:
        if rate >= 1_000_000:
            return f"{rate / 1_000_000:.2f}M"
        if rate >= 1_000:
            return f"{rate / 1_000:.1f}k"
        return f"{rate:.0f}"

    def _refresh(self) -> None:
        now = time.monotonic()
        with self.progress_lock:
            current = int(self.progress_bar.n)
            self.samples.append((now, current))
            cutoff = now - CURRENT_RATE_WINDOW_SECONDS
            while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
                self.samples.popleft()
            started, initial = self.samples[0]
            rate = max(current - initial, 0) / max(now - started, 1e-9)
            remaining = max(int(self.progress_bar.total or current) - current, 0)
            eta = f", ETA {remaining / rate:.0f}s" if rate > 0 and remaining else ""
            self.progress_bar.set_postfix_str(
                f"15s {self._format_rate(rate)} est tok/s{eta}", refresh=True
            )

    def _run(self) -> None:
        while not self.stop_event.wait(1.0):
            self._refresh()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        self._refresh()

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
    # Bulk snapshots avoid the public APIs and per-document Gutenberg hosts.
    # They are materially faster, resumable at input-shard boundaries, and do
    # not consume shared IP quotas intended for interactive API clients.
    "gutenberg": {
        "variants": [
            {"path": "common-pile/project_gutenberg", "name": None, "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "metadata.url"),
        "url_fields": ("metadata.url",),
        "title_fields": ("metadata.title",),
        "license": "Public domain (per-document metadata applies)",
        "kind": "books",
    },
    "stackexchange": {
        "variants": [
            {"path": "common-pile/stackexchange_filtered", "name": None, "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "metadata.url"),
        "url_fields": ("metadata.url",),
        "title_fields": (),
        "license": "CC BY-SA (per-document metadata applies)",
        "kind": "technical",
    },
    "arxiv": {
        "variants": [
            {"path": "common-pile/arxiv_abstracts_filtered", "name": None, "split": "train"},
        ],
        "text_fields": ("text",),
        "id_fields": ("id", "metadata.url"),
        "url_fields": ("metadata.url",),
        "title_fields": ("metadata.title",),
        "license": "Per-document arXiv/Common Pile metadata applies",
        "kind": "technical",
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
            response = _http_session().get(url, params=params, timeout=timeout)
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
        value = record
        for component in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(component)
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


def progress_reaches_budget(progress: dict, budget: SourceBudget) -> bool:
    """Check a saved source without constructing its potentially huge indexes."""
    token_target = budget.target_tokens_estimate
    return bool(
        (budget.target_docs > 0 and int(progress.get("docs_written", 0)) >= budget.target_docs)
        or (
            token_target > 0
            and int(progress.get("estimated_tokens", 0)) >= token_target
        )
        or (
            token_target <= 0
            and budget.target_chars > 0
            and int(progress.get("chars_written", 0)) >= budget.target_chars
        )
    )


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
        self.handle = self.current_path.open(
            "a", encoding="utf-8", buffering=JSONL_BUFFER_BYTES
        )
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
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
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

    def flush(self) -> None:
        if self.handle is not None:
            self.handle.flush()

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
        self.hf_dataset = None
        # Shared across all concurrently running sources: one overall bar,
        # updated under a lock since tqdm's counter isn't safe for concurrent
        # writers from multiple threads.
        self.progress_bar = progress_bar
        self.progress_lock = progress_lock

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
        # The progress cursor must never get ahead of buffered JSONL output.
        self.writer.flush()
        self._flush_seen(self.seen_ids_path, self.new_seen_ids)
        self._flush_seen(self.seen_urls_path, self.new_seen_urls)
        self._flush_seen(self.seen_titles_path, self.new_seen_titles)
        if self.hf_dataset is not None:
            self.progress["hf_stream_state"] = self.hf_dataset.state_dict()
        self._refresh_progress_metadata(self.progress)
        with self.progress_path.open("w", encoding="utf-8") as handle:
            json.dump(self.progress, handle, ensure_ascii=False, indent=2)

    def _flush_seen(self, path: Path, values: set[str]) -> None:
        if not values:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(sorted(values)) + "\n")
        values.clear()

    def maybe_save(self, min_interval: float = 10.0) -> None:
        """Persist progress at most once per interval to avoid disk churn."""
        now = time.monotonic()
        if now - self.last_save_time >= min_interval:
            self.save()
            self.last_save_time = now

    def should_stop(self) -> bool:
        if STOP_EVENT.is_set():
            return True
        return progress_reaches_budget(self.progress, self.budget)

    def identity_keys(
        self, doc_id: object = "", title: object = "", url: object = ""
    ) -> tuple[str, str, str]:
        return (
            self._seen_key(str(doc_id or ""), namespace="id"),
            self._seen_key(str(title or "").strip().lower(), namespace="title"),
            self._seen_key(str(url or "").strip().lower(), namespace="url"),
        )

    def has_seen_identity(
        self, doc_id: object = "", title: object = "", url: object = ""
    ) -> bool:
        return self.has_seen_identity_keys(self.identity_keys(doc_id, title, url))

    def has_seen_identity_keys(self, keys: tuple[str, str, str]) -> bool:
        doc_id_key, title_key, url_key = keys
        return bool(
            (doc_id_key and doc_id_key in self.seen_ids)
            or (title_key and title_key in self.seen_titles)
            or (url_key and url_key in self.seen_urls)
        )

    def accept(self, record: dict, english_only: bool) -> bool:
        text = normalize_text(record.get("text", ""))
        if len(text) < MIN_DOCUMENT_CHARS or looks_navigation_heavy(text):
            return False
        if english_only and not is_probably_english(text, self.config):
            return False

        doc_id = str(record.get("id", "") or "").strip()
        title = str(record.get("title", "") or "").strip().lower()
        url = str(record.get("url", "") or "").strip().lower()
        doc_id_key, title_key, url_key = self.identity_keys(doc_id, title, url)
        if self.has_seen_identity_keys((doc_id_key, title_key, url_key)):
            return False

        record["text"] = text
        self.writer.write(record)
        token_delta = max(1, len(text) // 4)
        self.progress["docs_written"] += 1
        self.progress["chars_written"] += len(text)
        self.progress["estimated_tokens"] += token_delta
        if self.progress_bar is not None and self.progress_lock is not None:
            # tqdm itself rate-limits redraws. Updating the true accepted-token
            # counter here avoids delayed batches being mistaken for download
            # throughput; the bar is explicitly labelled as accepted tokens.
            with self.progress_lock:
                self.progress_bar.update(token_delta)
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


def load_hf_stream(source_name: str, preferred_variant: int | None = None):
    spec = HF_DATASETS[source_name]
    last_error = None
    variant_indices = (
        [preferred_variant]
        if preferred_variant is not None and 0 <= preferred_variant < len(spec["variants"])
        else list(range(len(spec["variants"])))
    )
    for variant_index in variant_indices:
        variant = spec["variants"][variant_index]
        try:
            kwargs = {
                "path": variant["path"],
                "split": variant.get("split", "train"),
                "streaming": True,
            }
            if variant.get("name"):
                kwargs["name"] = variant["name"]
            dataset = load_dataset(**kwargs)
            return dataset, variant, variant_index
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to stream dataset for {source_name}: {last_error}")


def materialize_python_edu_row(row: dict) -> dict | None:
    """Fetch the file body referenced by a SmolLM Python-Edu metadata row."""
    blob_id = str(row.get("blob_id", "")).strip()
    if not blob_id:
        return None
    url = f"{SOFTWARE_HERITAGE_CONTENT_URL}/{blob_id}"
    last_error: Exception | None = None
    for attempt in range(5):
        if STOP_EVENT.is_set():
            return None
        try:
            with _http_session().get(url, timeout=(10, 60)) as response:
                if response.status_code == 404:
                    return None
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = requests.HTTPError(
                        f"{response.status_code} {response.reason} for {url}",
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    text = gzip.decompress(response.content).decode(
                        "utf-8", errors="ignore"
                    )
                    materialized = dict(row)
                    materialized["text"] = text
                    return materialized
        except (requests.RequestException, EOFError, gzip.BadGzipFile) as exc:
            last_error = exc

        if attempt < 4 and not STOP_EVENT.wait(min(2 ** attempt, 8)):
            continue
        break

    if STOP_EVENT.is_set():
        return None
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed request to {url}")


def format_hf_record(source_name: str, row: dict, variant: dict) -> dict | None:
    spec = HF_DATASETS[source_name]
    text = pick_first(row, spec["text_fields"])
    if not text:
        return None
    title = pick_first(row, spec["title_fields"])
    url = pick_first(row, spec["url_fields"])
    if source_name == "wikipedia_snapshot" and title and not url:
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    upstream_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    doc_id = pick_first(row, spec["id_fields"]) or url or title
    if not doc_id:
        doc_id = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return {
        "id": doc_id,
        "title": title,
        "url": url,
        "text": text,
        "meta": {
            "source": source_name,
            "dataset": variant["path"],
            "config": variant.get("name"),
            "license": upstream_metadata.get("license", spec["license"]),
        },
    }


def _iter_batches(values, size: int):
    iterator = iter(values)
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch


class DaemonOrderedMapper:
    """Reuse daemon fetch workers while yielding results in input order."""

    def __init__(self, function: Callable, workers: int, name: str = "item-fetch"):
        self.function = function
        self.stop_event = threading.Event()
        self.tasks: queue.Queue[tuple[int, object] | None] = queue.Queue()
        self.results: queue.Queue[tuple[int, object, Exception | None]] = queue.Queue()
        self.threads = [
            threading.Thread(
                target=self._work,
                name=f"{name}-{index}",
                daemon=True,
            )
            for index in range(max(1, workers))
        ]
        for thread in self.threads:
            thread.start()

    def _work(self) -> None:
        while not self.stop_event.is_set() and not STOP_EVENT.is_set():
            task = self.tasks.get()
            if task is None:
                return
            index, value = task
            try:
                self.results.put((index, self.function(value), None))
            except Exception as exc:
                self.results.put((index, None, exc))

    def map_ordered(self, values: list[dict]):
        for index, value in enumerate(values):
            self.tasks.put((index, value))

        buffered: dict[int, tuple[object, Exception | None]] = {}
        next_index = 0
        while next_index < len(values):
            if STOP_EVENT.is_set() or self.stop_event.is_set():
                return
            if next_index not in buffered:
                try:
                    index, value, error = self.results.get(timeout=0.2)
                except queue.Empty:
                    continue
                buffered[index] = (value, error)
                continue
            value, error = buffered.pop(next_index)
            if error is not None:
                raise error
            yield value
            next_index += 1

    def close(self) -> None:
        self.stop_event.set()
        for _ in self.threads:
            self.tasks.put(None)
        # Normal completion joins immediately. On Ctrl+C or a stuck request,
        # cap the total wait because these are daemon threads by design.
        deadline = time.monotonic() + 0.5
        for thread in self.threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


@dataclass
class _HFStreamItem:
    worker_index: int
    row: dict | None = None
    stream_state: dict | None = None
    error: Exception | None = None
    done: bool = False


def iter_parallel_hf_rows(dataset, state: SourceState, workers: int):
    """Merge independent HF input-shard streams without losing resume state."""
    worker_count = min(max(1, workers), int(dataset.num_shards))
    saved_count = int(state.progress.get("hf_parallel_workers", worker_count))
    if state.progress.get("hf_parallel_states") and saved_count != worker_count:
        worker_count = saved_count
    saved_states = list(state.progress.get("hf_parallel_states", [None] * worker_count))
    if len(saved_states) != worker_count:
        raise RuntimeError("HF parallel resume state does not match its saved worker count")
    state.progress["hf_parallel_workers"] = worker_count
    state.progress["hf_parallel_states"] = saved_states

    # Keep enough decoded rows ready that local filtering/writes don't make the
    # network producers repeatedly go idle. The cap remains modest for sources
    # with very large documents such as Gutenberg.
    output: queue.Queue[_HFStreamItem] = queue.Queue(
        maxsize=min(64, max(8, worker_count * 8))
    )
    stop = threading.Event()

    def emit(item: _HFStreamItem) -> bool:
        while not stop.is_set() and not STOP_EVENT.is_set():
            try:
                output.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce(worker_index: int) -> None:
        try:
            stream = dataset.shard(worker_count, worker_index, contiguous=False)
            if saved_states[worker_index]:
                stream.load_state_dict(saved_states[worker_index])
            for row in stream:
                if stop.is_set() or STOP_EVENT.is_set():
                    break
                if not emit(_HFStreamItem(worker_index, row=row, stream_state=stream.state_dict())):
                    break
        except Exception as exc:
            emit(_HFStreamItem(worker_index, error=exc))
        finally:
            emit(_HFStreamItem(worker_index, done=True))

    threads = [
        threading.Thread(target=produce, args=(index,), name=f"hf-shard-{index}", daemon=True)
        for index in range(worker_count)
    ]
    for thread in threads:
        thread.start()
    completed = 0
    try:
        while completed < worker_count:
            item = output.get()
            if item.error is not None:
                raise item.error
            if item.done:
                completed += 1
                continue
            assert item.row is not None and item.stream_state is not None
            yield item.row, item.worker_index, item.stream_state
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)


def ingest_hf_source(
    source_name: str,
    state: SourceState,
    english_only: bool,
    item_workers: int = DEFAULT_ITEM_WORKERS,
    hf_workers: int = DEFAULT_HF_WORKERS,
) -> None:
    if state.should_stop():
        state.save()
        log(f"  {source_name}: already at target, skipping download")
        return
    preferred_variant = state.progress.get("hf_variant_index")
    dataset, variant, variant_index = load_hf_stream(source_name, preferred_variant)
    state.progress["hf_variant_index"] = variant_index

    saved_stream_state = state.progress.get("hf_stream_state")
    rows_seen = int(state.progress.get("hf_rows_seen", 0))
    parallel_state = state.progress.get("hf_parallel_states")
    if saved_stream_state:
        dataset.load_state_dict(saved_stream_state)
    elif rows_seen > 0 and not parallel_state:
        # Compatibility for progress files created before datasets exposed a
        # serializable stream cursor. This replay happens once; all subsequent
        # saves use load_state_dict() and resume directly at the input shard.
        log(f"  {source_name}: migrating legacy row cursor ({rows_seen:,} rows; one-time replay)")
        for _ in itertools.islice(dataset, rows_seen):
            if STOP_EVENT.is_set():
                return

    if source_name == "python_edu":
        state.hf_dataset = dataset
        pending_rows = list(state.progress.get("hf_pending_rows", []))
        source_rows = itertools.chain(pending_rows, dataset)
        batch_size = max(1, item_workers * PYTHON_EDU_FETCH_AHEAD)
        with DaemonOrderedMapper(
            materialize_python_edu_row,
            item_workers,
            name="python-edu-fetch",
        ) as mapper:
            for batch in _iter_batches(source_rows, batch_size):
                # The stream cursor may already point beyond this prefetched
                # batch. Persist uncommitted metadata so Ctrl+C cannot create a
                # gap, even though content fetches happen concurrently.
                state.progress["hf_pending_rows"] = batch

                scheduled_ids: set[str] = set()
                fetch_flags: list[bool] = []
                fetch_rows: list[dict] = []
                for row in batch:
                    length_bytes = row.get("length_bytes")
                    too_short = False
                    if length_bytes is not None:
                        try:
                            too_short = int(length_bytes) < MIN_DOCUMENT_CHARS
                        except (TypeError, ValueError):
                            pass

                    blob_id = str(row.get("blob_id", "") or "").strip()
                    keys = state.identity_keys(blob_id, row.get("path", ""), "")
                    duplicate = state.has_seen_identity_keys(keys)
                    if keys[0] and keys[0] in scheduled_ids:
                        duplicate = True
                    should_fetch = bool(blob_id) and not too_short and not duplicate
                    fetch_flags.append(should_fetch)
                    if should_fetch:
                        scheduled_ids.add(keys[0])
                        fetch_rows.append(row)

                fetched_rows = iter(mapper.map_ordered(fetch_rows))
                for index, should_fetch in enumerate(fetch_flags):
                    if state.should_stop():
                        state.progress["hf_pending_rows"] = batch[index:]
                        return
                    if should_fetch:
                        try:
                            row = next(fetched_rows)
                        except StopIteration:
                            # Ctrl+C stops the mapper before it yields the rest;
                            # retain the uncommitted suffix for exact resume.
                            state.progress["hf_pending_rows"] = batch[index:]
                            return
                    else:
                        row = None
                        state.progress["python_edu_prefiltered_rows"] = (
                            int(state.progress.get("python_edu_prefiltered_rows", 0)) + 1
                        )

                    state.progress["hf_rows_seen"] += 1
                    state.progress["hf_pending_rows"] = batch[index + 1:]
                    if row is not None:
                        record = format_hf_record(source_name, row, variant)
                        if record is not None:
                            state.accept(record, english_only=english_only)
                    state.maybe_save()
                state.progress.pop("hf_pending_rows", None)
        return

    if not saved_stream_state and (parallel_state or (rows_seen <= 0 and hf_workers > 1)):
        parallel_rows = iter_parallel_hf_rows(dataset, state, hf_workers)
        try:
            while not state.should_stop():
                try:
                    row, worker_index, stream_state = next(parallel_rows)
                except StopIteration:
                    break
                state.progress["hf_rows_seen"] += 1
                record = format_hf_record(source_name, row, variant)
                if record is not None:
                    state.accept(record, english_only=english_only)
                state.progress["hf_parallel_states"][worker_index] = stream_state
                state.maybe_save()
        finally:
            parallel_rows.close()
        return

    state.hf_dataset = dataset
    row_iterator = iter(dataset)
    while not state.should_stop():
        try:
            row = next(row_iterator)
        except StopIteration:
            break
        state.progress["hf_rows_seen"] += 1
        record = format_hf_record(source_name, row, variant)
        if record is not None:
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
    item_workers: int = DEFAULT_ITEM_WORKERS,
    hf_workers: int = DEFAULT_HF_WORKERS,
) -> int:
    """Download a single source end to end. Runs in its own worker thread."""
    source_dir = Path(config.large_corpus_dir) / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    state = SourceState(
        source_dir, budget, resume=resume, config=config,
        progress_bar=progress_bar, progress_lock=progress_lock,
    )
    try:
        if source_name in HF_DATASETS:
            ingest_hf_source(
                source_name,
                state,
                english_only=english_only,
                item_workers=item_workers,
                hf_workers=hf_workers,
            )
        else:
            SOURCE_HANDLERS[source_name](state, english_only)
    finally:
        state.close()
    log(f"  completed [{source_name}]: {state.progress_summary()}")
    return int(state.progress["estimated_tokens"])


def read_source_progress(root: Path, source_name: str) -> dict:
    path = root / source_name / "progress.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_direct_hf_cursor(progress: dict) -> bool:
    return bool(progress.get("hf_stream_state") or progress.get("hf_parallel_states"))


def should_defer_legacy_cursor(progress: dict, budget: SourceBudget) -> bool:
    """Avoid replaying millions of rows to fill a negligible old-run tail.

    The missing tokens remain part of the global target and are filled by a
    source with a direct stream cursor during shortfall redistribution.
    """
    if int(progress.get("hf_rows_seen", 0)) <= 0 or has_direct_hf_cursor(progress):
        return False
    remaining = max(
        budget.target_tokens_estimate - int(progress.get("estimated_tokens", 0)), 0
    )
    threshold = max(
        LEGACY_CURSOR_DEFER_MIN_TOKENS,
        int(budget.target_tokens_estimate * LEGACY_CURSOR_DEFER_FRACTION),
    )
    return 0 < remaining <= threshold


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a large pretraining corpus into JSONL shards")
    parser.add_argument("--sources", type=str, default="all", help="Comma-separated list of sources")
    parser.add_argument("--target_tokens_estimate", type=int, default=0, help="Estimated processed token target")
    parser.add_argument("--max_docs", type=int, default=0, help="Optional per-source document cap")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from progress manifests; without this flag, all existing "
            "files for the selected sources are deleted before downloading"
        ),
    )
    parser.add_argument("--reset_source", type=str, default="", help="Comma-separated list of sources to reset")
    parser.add_argument(
        "--english-only", "--english_only", dest="english_only", action="store_true",
        help="Keep only likely English documents",
    )
    parser.add_argument("--workers", type=int, default=0, help="Concurrent source downloads (0 = one per source)")
    parser.add_argument(
        "--item-workers",
        type=int,
        default=DEFAULT_ITEM_WORKERS,
        help=(
            "Concurrent per-document fetches for sources whose rows reference "
            f"external content (default: {DEFAULT_ITEM_WORKERS})"
        ),
    )
    parser.add_argument(
        "--hf-workers-per-source",
        type=int,
        default=DEFAULT_HF_WORKERS,
        help=(
            "Parallel Hugging Face input-shard streams kept available as other "
            f"sources finish (default: {DEFAULT_HF_WORKERS})"
        ),
    )
    parser.add_argument(
        "--no-redistribute-shortfall",
        action="store_true",
        help="Do not fill exhausted/failed source budgets from another requested streaming source",
    )
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
        if not args.resume or source_name in reset_sources:
            reset_source_dir(source_dir)
        budgets[source_name] = build_budget(source_name, source_plan[source_name], args.max_docs)

    total_target_tokens = sum(b.target_tokens_estimate for b in budgets.values())
    progress_by_source = {
        name: read_source_progress(root, name) if args.resume else {}
        for name in sources
    }
    already_done = sum(int(progress.get("estimated_tokens", 0)) for progress in progress_by_source.values())
    completed_sources = {
        name
        for name in sources
        if args.resume and progress_reaches_budget(progress_by_source[name], budgets[name])
    }
    deferred_legacy_sources = {
        name
        for name in sources
        if name in HF_DATASETS
        and name not in completed_sources
        and args.resume
        and not args.no_redistribute_shortfall
        and should_defer_legacy_cursor(progress_by_source[name], budgets[name])
    }
    active_sources = [
        name
        for name in sources
        if name not in completed_sources and name not in deferred_legacy_sources
    ]
    workers = args.workers if args.workers > 0 else len(active_sources)
    workers = min(workers, len(active_sources)) if active_sources else 0
    log(f"Downloading {len(active_sources)} source(s) with {workers} concurrent worker(s)")
    if completed_sources:
        log(f"  already complete: {', '.join(sorted(completed_sources))}")
    for name in sorted(deferred_legacy_sources):
        progress = progress_by_source[name]
        remaining = max(
            budgets[name].target_tokens_estimate - int(progress.get("estimated_tokens", 0)), 0
        )
        log(
            f"  {name}: skipping {int(progress.get('hf_rows_seen', 0)):,}-row legacy replay; "
            f"its {remaining:,}-token tail will be filled from a direct-cursor source"
        )

    # Load the langid model only when work remains, and once up front so source
    # workers share it instead of racing on their first document.
    if args.english_only and active_sources:
        load_langid_model(config, logger=log)

    progress_lock = threading.Lock()
    progress_bar = tqdm(
        total=total_target_tokens or None,
        initial=already_done,
        desc="Accepted corpus",
        unit=" est tok",
        unit_scale=True,
        # Omit tqdm's update-triggered rate: it remains stale during retries or
        # cursor work. AcceptedRateMonitor includes zero-progress wall time.
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
    )
    rate_monitor = AcceptedRateMonitor(progress_bar, progress_lock)

    failures: list[tuple[str, str]] = []
    interrupted = False
    try:
        source_results: queue.Queue[tuple[str, Exception | None]] = queue.Queue()
        source_tasks: queue.Queue[str] = queue.Queue()
        for source_name in active_sources:
            source_tasks.put(source_name)

        def execute_sources() -> None:
            while not STOP_EVENT.is_set():
                try:
                    name = source_tasks.get_nowait()
                except queue.Empty:
                    return
                error = None
                try:
                    run_source(
                        name, budgets[name], config, args.english_only, args.resume,
                        progress_bar, progress_lock, args.item_workers, args.hf_workers_per_source,
                    )
                except Exception as exc:
                    error = exc
                finally:
                    source_results.put((name, error))

        source_threads = [
            threading.Thread(
                target=execute_sources, name=f"source-worker-{index}", daemon=True
            )
            for index in range(workers)
        ]
        pending_sources = set(active_sources)
        for thread in source_threads:
            thread.start()
        try:
            while pending_sources:
                try:
                    source_name, exc = source_results.get(timeout=0.2)
                except queue.Empty:
                    continue
                pending_sources.discard(source_name)
                if exc is not None:
                    failures.append((source_name, str(exc)))
                    log(f"  source failed: {source_name} -> {exc}")
        except KeyboardInterrupt:
            interrupted = True
            STOP_EVENT.set()
            log(
                f"\nInterrupted; allowing workers up to "
                f"{INTERRUPT_GRACE_SECONDS:g} seconds to flush progress..."
            )
            deadline = time.monotonic() + INTERRUPT_GRACE_SECONDS
            try:
                while pending_sources and time.monotonic() < deadline:
                    try:
                        source_name, exc = source_results.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    pending_sources.discard(source_name)
                    if exc is not None:
                        failures.append((source_name, str(exc)))
            except KeyboardInterrupt:
                # A second Ctrl+C skips the remaining grace period.
                pass
            if pending_sources:
                log(
                    f"  exiting with the latest saved checkpoints; "
                    f"{len(pending_sources)} network worker(s) are still blocked"
                )

        if not interrupted and not args.no_redistribute_shortfall and args.max_docs <= 0:
            accepted_tokens = sum(
                int(json.loads((root / name / "progress.json").read_text()).get("estimated_tokens", 0))
                for name in sources
                if (root / name / "progress.json").exists()
            )
            shortfall = max(total_target_tokens - accepted_tokens, 0)
            fallback_order = (
                "stackexchange", "arxiv", "gutenberg", "openwebmath", "finemath",
                "fineweb_sample", "wikipedia_snapshot", "cosmopedia_v2", "refinedweb",
                "c4_en", "fineweb-edu", "python_edu",
            )
            eligible_fallbacks = [
                name for name in fallback_order
                if name in sources and name in HF_DATASETS and name not in deferred_legacy_sources
            ]
            fallback = next(
                (
                    name for name in eligible_fallbacks
                    if has_direct_hf_cursor(read_source_progress(root, name))
                ),
                eligible_fallbacks[0] if eligible_fallbacks else None,
            )
            if shortfall and fallback is not None:
                log(
                    f"  redistributing {shortfall:,} unfilled est tokens to {fallback}; "
                    "the requested global target remains unchanged"
                )
                budget = budgets[fallback]
                budget.target_tokens_estimate += shortfall
                budget.target_chars += shortfall * 4
                try:
                    run_source(
                        fallback, budget, config, args.english_only, True,
                        progress_bar, progress_lock, args.item_workers, args.hf_workers_per_source,
                    )
                except Exception as exc:
                    failures.append((f"{fallback} (shortfall fill)", str(exc)))
                    log(f"  shortfall fill failed: {fallback} -> {exc}")
    finally:
        rate_monitor.close()
        progress_bar.close()

    if interrupted:
        return 130

    final_accepted_tokens = sum(
        int(json.loads((root / name / "progress.json").read_text()).get("estimated_tokens", 0))
        for name in sources
        if (root / name / "progress.json").exists()
    )
    target_reached = final_accepted_tokens >= total_target_tokens
    if failures:
        heading = "Recovered source failures" if target_reached else "Completed with source failures"
        log(f"\n{heading}:")
        for source_name, message in failures:
            log(f"  - {source_name}: {message}")
    if (failures or args.max_docs <= 0) and not target_reached:
        log(
            f"\nCorpus target not reached: {final_accepted_tokens:,} / "
            f"{total_target_tokens:,} est tokens"
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        STOP_EVENT.set()
        log("\nInterrupted while downloading pretraining data.")
        sys.exit(130)
