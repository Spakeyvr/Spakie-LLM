"""Scrape open-access text corpora from several permissive platforms.

Sources:
- Project Gutenberg via Gutendex API (public domain books)
- arXiv API (abstracts for recent papers)
- Stack Exchange API (high-score Stack Overflow Q&A, CC BY-SA)

All documents are saved as markdown files under data/raw/open_corpus/<source>/.
Progress is tracked so the script can be interrupted and resumed.

Usage:
    python scripts/scrape_open_corpus.py
    python scripts/scrape_open_corpus.py --gutenberg 12 --arxiv 90 --stackoverflow 80
    python scripts/scrape_open_corpus.py --reset
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests


BASE_DIR = Path("data/raw/open_corpus")
PROGRESS_FILE = BASE_DIR / ".progress.json"
HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}

GUTENDEX_API = "https://gutendex.com/books"
ARXIV_API = "https://export.arxiv.org/api/query"
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"

ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "math.ST",
    "stat.ML",
]


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def sanitize_filename(name: str, max_len: int = 140) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    if not name:
        name = "document"
    return name[:max_len]


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def write_markdown(source: str, stem: str, body: str) -> Path:
    source_dir = BASE_DIR / source
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{sanitize_filename(stem)}.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(body.strip() + "\n")
    return path


def default_progress() -> dict:
    return {
        "gutenberg_ids": [],
        "arxiv_ids": [],
        "stackoverflow_ids": [],
    }


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    return normalize_whitespace(text[start:end])


def choose_text_url(formats: dict[str, str]) -> str | None:
    preferred = [
        "text/plain; charset=utf-8",
        "text/plain",
        "text/plain; charset=us-ascii",
    ]
    for key in preferred:
        url = formats.get(key)
        if url and not url.endswith(".zip"):
            return url

    for key, url in formats.items():
        if key.startswith("text/plain") and not url.endswith(".zip"):
            return url
    return None


def scrape_gutenberg(target_count: int, progress: dict) -> int:
    done_ids = set(progress["gutenberg_ids"])
    scraped = 0
    next_url = GUTENDEX_API

    print(f"\n[Gutenberg] Target: {target_count}")
    while next_url and scraped < target_count:
        response = api_get(next_url)
        payload = response.json()

        for book in payload.get("results", []):
            book_id = book["id"]
            if book_id in done_ids:
                continue

            if "en" not in book.get("languages", []):
                continue

            text_url = choose_text_url(book.get("formats", {}))
            if not text_url:
                continue

            try:
                book_response = api_get(text_url, timeout=120)
            except Exception as exc:
                print(f"  Skip Gutenberg #{book_id}: {exc}")
                done_ids.add(book_id)
                progress["gutenberg_ids"].append(book_id)
                save_json(PROGRESS_FILE, progress)
                continue

            raw_text = book_response.text
            text = strip_gutenberg_boilerplate(raw_text)
            if len(text) < 20_000:
                done_ids.add(book_id)
                progress["gutenberg_ids"].append(book_id)
                save_json(PROGRESS_FILE, progress)
                continue

            authors = ", ".join(author["name"] for author in book.get("authors", [])) or "Unknown"
            markdown = (
                f"# {book['title']}\n\n"
                f"Source: Project Gutenberg\n"
                f"Authors: {authors}\n"
                f"Book ID: {book_id}\n"
                f"Downloads: {book.get('download_count', 0)}\n\n"
                f"{text}\n"
            )
            write_markdown("gutenberg", f"{book_id}_{book['title']}", markdown)

            done_ids.add(book_id)
            progress["gutenberg_ids"].append(book_id)
            scraped += 1
            save_json(PROGRESS_FILE, progress)
            print(f"  [{scraped}/{target_count}] {book['title']} ({len(text):,} chars)")

            if scraped >= target_count:
                break
            time.sleep(1)

        next_url = payload.get("next")

    return scraped


def parse_arxiv_feed(xml_text: str) -> list[dict]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    items = []
    for entry in root.findall("atom:entry", ns):
        item_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
        title = normalize_whitespace(entry.findtext("atom:title", default="", namespaces=ns))
        summary = normalize_whitespace(entry.findtext("atom:summary", default="", namespaces=ns))
        published = entry.findtext("atom:published", default="", namespaces=ns).strip()
        authors = [
            normalize_whitespace(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        categories = [cat.attrib.get("term", "") for cat in entry.findall("atom:category", ns)]
        if item_id and title and summary:
            items.append(
                {
                    "id": item_id,
                    "title": title,
                    "summary": summary,
                    "published": published,
                    "authors": authors,
                    "categories": categories,
                }
            )
    return items


def scrape_arxiv(target_count: int, progress: dict) -> int:
    done_ids = set(progress["arxiv_ids"])
    scraped = 0
    seen_ids = set(done_ids)
    per_category = max(10, target_count // max(1, len(ARXIV_CATEGORIES)) + 5)

    print(f"\n[arXiv] Target: {target_count}")
    queue: list[dict] = []
    for category in ARXIV_CATEGORIES:
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": per_category,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = api_get(ARXIV_API, params=params)
        queue.extend(parse_arxiv_feed(response.text))
        time.sleep(1)

    random.shuffle(queue)

    for item in queue:
        if scraped >= target_count:
            break
        if item["id"] in seen_ids:
            continue

        content = (
            f"# {item['title']}\n\n"
            f"Source: arXiv\n"
            f"Published: {item['published']}\n"
            f"Authors: {', '.join(item['authors'])}\n"
            f"Categories: {', '.join(item['categories'])}\n"
            f"Paper ID: {item['id']}\n\n"
            f"## Abstract\n\n"
            f"{item['summary']}\n"
        )
        write_markdown("arxiv", item["title"], content)
        seen_ids.add(item["id"])
        progress["arxiv_ids"].append(item["id"])
        scraped += 1
        save_json(PROGRESS_FILE, progress)
        print(f"  [{scraped}/{target_count}] {item['title']}")

    return scraped


def batched(values: Iterable[int], size: int) -> Iterable[list[int]]:
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_answers(answer_ids: list[int]) -> dict[int, dict]:
    answers = {}
    for batch in batched(answer_ids, 20):
        ids = ";".join(str(value) for value in batch)
        url = f"{STACKEXCHANGE_API}/answers/{ids}"
        params = {
            "site": "stackoverflow",
            "filter": "withbody",
            "pagesize": len(batch),
        }
        response = api_get(url, params=params)
        for item in response.json().get("items", []):
            answers[item["answer_id"]] = item
        time.sleep(0.5)
    return answers


def scrape_stackoverflow(target_count: int, progress: dict) -> int:
    done_ids = set(progress["stackoverflow_ids"])
    scraped = 0
    page = 1

    print(f"\n[Stack Overflow] Target: {target_count}")
    while scraped < target_count:
        params = {
            "site": "stackoverflow",
            "page": page,
            "pagesize": 100,
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
        }
        response = api_get(f"{STACKEXCHANGE_API}/questions", params=params)
        payload = response.json()
        questions = payload.get("items", [])
        if not questions:
            break

        accepted_ids = [
            question["accepted_answer_id"]
            for question in questions
            if question.get("accepted_answer_id") and question["question_id"] not in done_ids
        ]
        answers = fetch_answers(accepted_ids)

        for question in questions:
            question_id = question["question_id"]
            if question_id in done_ids:
                continue
            answer_id = question.get("accepted_answer_id")
            answer = answers.get(answer_id)
            if not answer:
                continue

            question_text = html_to_text(question.get("body", ""))
            answer_text = html_to_text(answer.get("body", ""))
            if len(question_text) < 120 or len(answer_text) < 200:
                done_ids.add(question_id)
                progress["stackoverflow_ids"].append(question_id)
                save_json(PROGRESS_FILE, progress)
                continue

            tags = ", ".join(question.get("tags", []))
            markdown = (
                f"# {html.unescape(question['title'])}\n\n"
                f"Source: Stack Overflow\n"
                f"Question ID: {question_id}\n"
                f"Accepted Answer ID: {answer_id}\n"
                f"Score: {question.get('score', 0)}\n"
                f"Answer Score: {answer.get('score', 0)}\n"
                f"Tags: {tags}\n\n"
                f"## Question\n\n"
                f"{question_text}\n\n"
                f"## Accepted Answer\n\n"
                f"{answer_text}\n"
            )
            write_markdown("stackoverflow", f"{question_id}_{question['title']}", markdown)

            done_ids.add(question_id)
            progress["stackoverflow_ids"].append(question_id)
            scraped += 1
            save_json(PROGRESS_FILE, progress)
            print(f"  [{scraped}/{target_count}] {html.unescape(question['title'])}")

            if scraped >= target_count:
                break
            time.sleep(0.5)

        if not payload.get("has_more"):
            break
        page += 1
        time.sleep(1)

    return scraped


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape open text corpora into markdown files")
    parser.add_argument("--gutenberg", type=int, default=12, help="Number of Gutenberg books to fetch")
    parser.add_argument("--arxiv", type=int, default=90, help="Number of arXiv abstracts to fetch")
    parser.add_argument("--stackoverflow", type=int, default=80, help="Number of Stack Overflow Q&A pairs to fetch")
    parser.add_argument("--reset", action="store_true", help="Delete progress and start over")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("Progress cleared.")

    progress = load_json(PROGRESS_FILE, default_progress())

    total_g = scrape_gutenberg(args.gutenberg, progress)
    total_a = scrape_arxiv(args.arxiv, progress)
    total_s = scrape_stackoverflow(args.stackoverflow, progress)

    print("\nDone.")
    print(f"  Gutenberg:      {total_g}")
    print(f"  arXiv:          {total_a}")
    print(f"  Stack Overflow: {total_s}")


if __name__ == "__main__":
    main()
