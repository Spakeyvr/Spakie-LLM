"""Wiktionary scraper: fetches word definitions from most-viewed to least-viewed.

Saves .md files to data/raw/dictionary/. Tracks progress so you can Ctrl+C and resume.

Usage:
    python scripts/scrape_dictionary.py
    python scripts/scrape_dictionary.py --max 5000
    python scripts/scrape_dictionary.py --reset
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WIKT_API = "https://en.wiktionary.org/w/api.php"
PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wiktionary/all-access"
HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}

OUTPUT_DIR = "data/raw/dictionary"
PROGRESS_FILE = "data/raw/dictionary/.dict_progress.json"
QUEUE_FILE = "data/raw/dictionary/.dict_queue.json"

SKIP_PREFIXES = ("Special:", "Wiktionary:", "File:", "Template:", "Help:",
                 "Category:", "Appendix:", "Module:", "MediaWiki:", "Talk:",
                 "User:", "Reconstruction:", "Thesaurus:", "Citations:")


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def api_get(url: str, params: dict | None = None, max_retries: int = 5) -> requests.Response:
    """GET with exponential backoff on rate limit or server errors."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = min(2 ** attempt * 5, 120)
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            print(f"  Rate limited ({resp.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def fetch_top_words(months: int = 12) -> list[str]:
    """Fetch top-viewed Wiktionary entries over the last N months."""
    print(f"Building word queue from {months} months of pageview data...")
    view_counts: dict[str, int] = {}

    now = datetime.now()
    for i in range(months):
        date = now - timedelta(days=30 * (i + 1))
        url = f"{PAGEVIEWS_API}/{date.year}/{date.month:02d}/all-days"
        try:
            resp = api_get(url)
            items = resp.json()["items"][0]["articles"]
            for item in items:
                title = item["article"]
                if title.startswith(("Wiktionary:Main_Page", "-")):
                    continue
                if any(title.startswith(p) for p in SKIP_PREFIXES):
                    continue
                # Skip non-word entries (numbers, single chars, etc.)
                if len(title) < 2 or title[0].isdigit():
                    continue
                view_counts[title] = view_counts.get(title, 0) + item["views"]
            print(f"  {date.year}-{date.month:02d}: got {len(items)} entries")
        except Exception as e:
            print(f"  {date.year}-{date.month:02d}: failed ({e})")
        time.sleep(0.5)

    ranked = sorted(view_counts.keys(), key=lambda t: view_counts[t], reverse=True)
    print(f"Total unique words in queue: {len(ranked)}")
    return ranked


def fetch_definition(title: str) -> str | None:
    """Fetch word definition as plain text from Wiktionary."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    try:
        resp = api_get(WIKT_API, params=params)
        pages = resp.json()["query"]["pages"]
        for page in pages.values():
            text = page.get("extract", "")
            if text and len(text) > 50:
                # Extract the English section if present
                english = extract_english_section(text)
                if english and len(english) > 50:
                    return english
                return text
    except Exception as e:
        print(f"  Failed to fetch '{title}': {e}")
    return None


def extract_english_section(text: str) -> str | None:
    """Pull out just the English section from a Wiktionary extract."""
    # Wiktionary entries are split by language headers like "== English =="
    parts = re.split(r"\n(?===\s)", text)
    for part in parts:
        if part.strip().startswith("== English") or part.strip().startswith("English"):
            return part.strip()
    # If no explicit English header, might be English-only — return as-is
    if "==" not in text[:100]:
        return text
    return None


def title_to_filename(title: str) -> str:
    name = title.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = re.sub(r'[<>"|?*]', "", name)
    name = name[:120]
    return name + ".md"


def format_as_markdown(title: str, text: str) -> str:
    display_title = title.replace("_", " ")
    return f"# {display_title}\n\n{text}\n"


def scrape(max_words: int = 0, reset: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if reset:
        for f in (PROGRESS_FILE, QUEUE_FILE):
            if os.path.exists(f):
                os.remove(f)
        print("Progress cleared.")

    queue = load_json(QUEUE_FILE, None)
    if queue is None:
        queue = fetch_top_words(months=12)
        save_json(QUEUE_FILE, queue)

    progress = load_json(PROGRESS_FILE, {"scraped": [], "index": 0, "retry": []})
    progress.setdefault("retry", [])
    scraped_set = set(progress["scraped"])
    start_index = progress["index"]

    print(f"\nResuming from index {start_index} / {len(queue)}")
    print(f"Already scraped: {len(scraped_set)} words")
    print("Press Ctrl+C to stop.\n")

    count = 0
    try:
        retry_titles = [title for title in progress["retry"] if title not in scraped_set]
        work = [(title, None) for title in retry_titles]
        work.extend((queue[i], i) for i in range(start_index, len(queue)))
        for title, queue_index in work:

            if title in scraped_set:
                if queue_index is not None:
                    progress["index"] = queue_index + 1
                continue

            text = fetch_definition(title)
            if text is None:
                # Advance the queue cursor so one bad entry does not block the
                # corpus, but retain an explicit retry queue. Only successful
                # writes enter `scraped`.
                if title not in progress["retry"]:
                    progress["retry"].append(title)
                if queue_index is not None:
                    progress["index"] = queue_index + 1
                save_json(PROGRESS_FILE, progress)
                continue

            filename = title_to_filename(title)
            filepath = os.path.join(OUTPUT_DIR, filename)
            md = format_as_markdown(title, text)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md)

            scraped_set.add(title)
            progress["scraped"].append(title)
            if title in progress["retry"]:
                progress["retry"].remove(title)
            if queue_index is not None:
                progress["index"] = queue_index + 1
            count += 1

            display = title.replace("_", " ")
            print(f"[{count:>5}] {display} ({len(text):,} chars)")

            save_json(PROGRESS_FILE, progress)

            if max_words and count >= max_words:
                print(f"\nReached max ({max_words}). Stopping.")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        save_json(PROGRESS_FILE, progress)
        print(f"\n\nStopped. Scraped {count} words this session.")
        print(f"Total scraped: {len(scraped_set)}")
        print("Run again to continue where you left off.")


def main():
    parser = argparse.ArgumentParser(description="Scrape Wiktionary definitions by popularity")
    parser.add_argument("--max", type=int, default=0, help="Max words to scrape (0 = unlimited)")
    parser.add_argument("--reset", action="store_true", help="Clear progress and start over")
    args = parser.parse_args()
    scrape(max_words=args.max, reset=args.reset)


if __name__ == "__main__":
    main()
