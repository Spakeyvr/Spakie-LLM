"""Download a large pretraining corpus into resumable JSONL shards.

The downloader keeps the existing training contract intact by writing raw text
under data/raw/large_corpus/<source>/ and leaving tokenization to
scripts/prepare_data.py.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

import requests
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig, normalize_corpus_source


HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}
GUTENDEX_API = "https://gutendex.com/books"
ARXIV_API = "https://export.arxiv.org/api/query"
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"

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


def api_get(url: str, *, params: dict | None = None, timeout: int = 60, max_retries: int = 5) -> requests.Response:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if response.status_code == 200:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                wait = min(2 ** attempt * 3, 60)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                print(f"  Retrying {url} after HTTP {response.status_code} ({wait}s)")
                time.sleep(wait)
                continue
            response.raise_for_status()
        except Exception as exc:
            last_error = exc
            wait = min(2 ** attempt * 2, 30)
            print(f"  Request failed: {exc} ({wait}s)")
            time.sleep(wait)
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


def is_probably_english(text: str) -> bool:
    sample = text[:3000]
    if not sample:
        return False
    letters = sum(ch.isalpha() for ch in sample)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in sample)
    spaces = sample.count(" ")
    if letters < 100 or spaces < 20:
        return False
    return ascii_letters / max(letters, 1) >= 0.75


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
    def __init__(self, source_dir: Path, budget: SourceBudget, resume: bool):
        self.source_dir = source_dir
        self.progress_path = source_dir / "progress.json"
        self.seen_ids_path = source_dir / "seen_ids.txt"
        self.seen_urls_path = source_dir / "seen_urls.txt"
        self.seen_titles_path = source_dir / "seen_titles.txt"
        self.budget = budget
        self.resume = resume
        self.progress = self._load_progress()
        self.writer = JsonlShardWriter(source_dir, budget.source_name, shard_char_limit=5_000_000, progress=self.progress)
        self.seen_ids = self._load_seen(self.seen_ids_path)
        self.seen_urls = self._load_seen(self.seen_urls_path)
        self.seen_titles = self._load_seen(self.seen_titles_path)
        self.new_seen_ids: set[str] = set()
        self.new_seen_urls: set[str] = set()
        self.new_seen_titles: set[str] = set()
        self.last_reported_docs = int(self.progress.get("docs_written", 0))

    def _load_seen(self, path: Path) -> set[str]:
        if self.resume and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return {line.strip() for line in handle if line.strip()}
        return set()

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

    def should_stop(self) -> bool:
        return (
            self.progress["chars_written"] >= self.budget.target_chars
            or self.progress["docs_written"] >= self.budget.target_docs
            or self.progress["estimated_tokens"] >= self.budget.target_tokens_estimate
        )

    def accept(self, record: dict, english_only: bool) -> bool:
        text = normalize_text(record.get("text", ""))
        if len(text) < 400 or looks_navigation_heavy(text):
            return False
        if english_only and not is_probably_english(text):
            return False

        doc_id = str(record.get("id", "")).strip()
        title = record.get("title", "").strip().lower()
        url = record.get("url", "").strip().lower()
        if doc_id and doc_id in self.seen_ids:
            return False
        if title and title in self.seen_titles:
            return False
        if url and url in self.seen_urls:
            return False

        record["text"] = text
        self.writer.write(record)
        self.progress["docs_written"] += 1
        self.progress["chars_written"] += len(text)
        self.progress["estimated_tokens"] += max(1, len(text) // 4)
        if doc_id:
            self.seen_ids.add(doc_id)
            self.new_seen_ids.add(doc_id)
        if title:
            self.seen_titles.add(title)
            self.new_seen_titles.add(title)
        if url:
            self.seen_urls.add(url)
            self.new_seen_urls.add(url)
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
    target_docs = max_docs if max_docs > 0 else max(2_000, target_chars // 8_000)
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
        print(f"  {source_name}: already at target, skipping download")
        return
    rows_seen = int(state.progress.get("hf_rows_seen", 0))
    record_iter = iter_hf_records(source_name)
    if rows_seen > 0:
        record_iter = itertools.islice(record_iter, rows_seen, None)
    for record in record_iter:
        state.progress["hf_rows_seen"] = int(state.progress.get("hf_rows_seen", 0)) + 1
        if state.should_stop():
            break
        accepted = state.accept(record, english_only=english_only)
        docs_written = int(state.progress["docs_written"])
        if accepted and docs_written % 100 == 0 and docs_written != state.last_reported_docs:
            state.save()
            print(f"  {source_name}: {state.progress_summary()}")
            state.last_reported_docs = docs_written
        if state.should_stop():
            break


def ingest_gutenberg(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        print("  gutenberg: already at target, skipping download")
        return
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
                raw_text = api_get(text_url, timeout=120).text
            except Exception as exc:
                print(f"  Skip Gutenberg #{book['id']}: {exc}")
                continue
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
            state.save()
            time.sleep(0.5)
        next_url = payload.get("next")


def ingest_arxiv(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        print("  arxiv: already at target, skipping download")
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
            state.save()
            print(f"  arxiv: {state.progress_summary()}")
            time.sleep(0.5)
        if not made_progress:
            break


def ingest_stackexchange(state: SourceState, english_only: bool) -> None:
    if state.should_stop():
        state.save()
        print("  stackexchange: already at target, skipping download")
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
            except Exception as exc:
                print(f"  Skip Stack Exchange site '{site}': {exc}")
                break
            questions = payload.get("items", [])
            if not questions:
                break
            accepted_ids = [q["accepted_answer_id"] for q in questions if q.get("accepted_answer_id")]
            answers = fetch_answers(accepted_ids, site)
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
            state.save()
            print(f"  stackexchange/{site}: {state.progress_summary()}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a large pretraining corpus into JSONL shards")
    parser.add_argument("--sources", type=str, default="all", help="Comma-separated list of sources")
    parser.add_argument("--target_tokens_estimate", type=int, default=0, help="Estimated processed token target")
    parser.add_argument("--max_docs", type=int, default=0, help="Optional per-source document cap")
    parser.add_argument("--resume", action="store_true", help="Resume from progress manifests")
    parser.add_argument("--reset_source", type=str, default="", help="Comma-separated list of sources to reset")
    parser.add_argument("--english_only", action="store_true", help="Keep only likely English documents")
    args = parser.parse_args()

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
    failures: list[tuple[str, str]] = []

    for source_name in sources:
        if source_name not in source_plan:
            raise ValueError(f"Unsupported source or disabled source plan entry: {source_name}")
        if source_name not in SOURCE_HANDLERS and source_name not in HF_DATASETS:
            raise ValueError(f"Unsupported source: {source_name}")

        source_dir = root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        if source_name in reset_sources:
            reset_source_dir(source_dir)

        budget = build_budget(source_name, source_plan[source_name], args.max_docs)
        state = SourceState(source_dir, budget, resume=args.resume)
        print(
            f"\n[{source_name}] kind={budget.kind} "
            f"target_chars={budget.target_chars:,} "
            f"target_docs={budget.target_docs:,} "
            f"target_tokens_est={budget.target_tokens_estimate:,}"
        )
        try:
            try:
                if source_name in HF_DATASETS:
                    ingest_hf_source(source_name, state, english_only=args.english_only)
                else:
                    SOURCE_HANDLERS[source_name](state, args.english_only)
            except Exception as exc:
                failures.append((source_name, str(exc)))
                print(f"  source failed: {source_name} -> {exc}")
        finally:
            state.close()
        print(f"  completed {state.progress_summary()}")

    if failures:
        print("\nCompleted with source failures:")
        for source_name, message in failures:
            print(f"  - {source_name}: {message}")


if __name__ == "__main__":
    main()
