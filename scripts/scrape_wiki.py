"""Wikipedia dump downloader + extractor.

Downloads the latest English Wikipedia articles-only dump (~22GB compressed),
streams and extracts articles into .md files in data/raw/wikipedia/.
Resumable — tracks progress so you can Ctrl+C and pick up where you left off.

Usage:
    python scripts/scrape_wiki.py                  # download + extract
    python scripts/scrape_wiki.py --max 10000      # stop after 10k articles
    python scripts/scrape_wiki.py --reset           # start over
"""

import argparse
import bz2
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig

DUMP_INDEX = "https://dumps.wikimedia.org/enwiki/latest/"
DUMP_FILE = "enwiki-latest-pages-articles.xml.bz2"
DUMP_URL = DUMP_INDEX + DUMP_FILE
HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}

OUTPUT_DIR = "data/raw/wikipedia"
DUMP_PATH = "data/raw/wikipedia/.dump.xml.bz2"
PROGRESS_FILE = "data/raw/wikipedia/.dump_progress.json"

# Namespaces to skip (only want namespace 0 = articles)
SKIP_TITLE_PREFIXES = (
    "Wikipedia:", "File:", "Template:", "Help:", "Category:", "Portal:",
    "Draft:", "Module:", "MediaWiki:", "Talk:", "User:", "User talk:",
    "Wikipedia talk:", "Template talk:", "Special:",
)


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def download_dump():
    """Download the Wikipedia dump with resume support."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if already downloaded
    downloaded = 0
    if os.path.exists(DUMP_PATH):
        downloaded = os.path.getsize(DUMP_PATH)

    # Get total size
    resp = requests.head(DUMP_URL, headers=HEADERS, allow_redirects=True, timeout=30)
    total_size = int(resp.headers.get("Content-Length", 0))

    if downloaded >= total_size and total_size > 0:
        print(f"Dump already downloaded ({total_size / 1e9:.1f} GB)")
        return

    print(f"Downloading Wikipedia dump ({total_size / 1e9:.1f} GB)...")
    if downloaded > 0:
        print(f"  Resuming from {downloaded / 1e9:.1f} GB")

    resume_headers = {**HEADERS}
    if downloaded > 0:
        resume_headers["Range"] = f"bytes={downloaded}-"

    resp = requests.get(DUMP_URL, headers=resume_headers, stream=True, timeout=30)

    mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
    if mode == "wb":
        downloaded = 0

    with open(DUMP_PATH, mode) as f:
        pbar = tqdm(total=total_size, initial=downloaded, unit="B", unit_scale=True, desc="Download")
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

    print("Download complete.")


def strip_wikitext(text: str) -> str:
    """Convert wikitext to plain text (best effort)."""
    # Remove XML/HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove {{ templates }}
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}" and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = "".join(result)
    # Convert [[links]]
    text = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)  # [[target|display]] -> display
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)             # [[article]] -> article
    # Remove [external links]
    text = re.sub(r"\[https?://[^\]]*\]", "", text)
    # Remove bold/italic markup
    text = re.sub(r"'{2,5}", "", text)
    # Convert headers
    text = re.sub(r"^={2,6}\s*(.+?)\s*={2,6}", r"## \1", text, flags=re.MULTILINE)
    # Remove remaining {| table markup
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def title_to_filename(title: str) -> str:
    name = title.replace("/", "_").replace("\\", "_").replace(":", "_")
    name = re.sub(r'[<>"|?*]', "", name)
    name = name[:120]
    return name + ".md"


def extract_articles(max_articles: int = 0, reset: bool = False):
    """Stream-parse the bz2 dump and extract articles to .md files."""
    if reset and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Progress cleared.")

    progress = load_json(PROGRESS_FILE, {"count": 0, "bytes_read": 0})
    start_count = progress["count"]
    skip_until = start_count

    print(f"\nExtracting articles from dump...")
    if start_count > 0:
        print(f"Skipping first {start_count} articles (already extracted)")
    print("Press Ctrl+C to stop.\n")

    count = 0
    extracted = 0

    try:
        with open(DUMP_PATH, "rb") as f:
            decompressor = bz2.BZ2Decompressor()
            parser = ET.XMLPullParser(["end"])
            leftover = b""

            # MediaWiki XML namespace
            ns = "{http://www.mediawiki.org/xml/export-0.11/}"

            title = None
            text = None
            in_page = False
            current_data = b""

            for raw_chunk in iter(lambda: f.read(1024 * 1024), b""):
                try:
                    data = decompressor.decompress(raw_chunk)
                except EOFError:
                    # Multi-stream bz2 — create a new decompressor
                    decompressor = bz2.BZ2Decompressor()
                    try:
                        data = decompressor.decompress(raw_chunk)
                    except EOFError:
                        continue

                parser.feed(data)

                for event, elem in parser.read_events():
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

                    if tag == "title":
                        title = elem.text or ""
                    elif tag == "text":
                        text = elem.text or ""
                    elif tag == "page":
                        count += 1

                        # Skip non-article pages
                        if title and not any(title.startswith(p) for p in SKIP_TITLE_PREFIXES):
                            if count <= skip_until:
                                elem.clear()
                                continue

                            if text and len(text) > 500:
                                # Skip redirects
                                if text.strip().upper().startswith("#REDIRECT"):
                                    elem.clear()
                                    continue

                                plain = strip_wikitext(text)
                                if len(plain) > 200:
                                    filename = title_to_filename(title)
                                    filepath = os.path.join(OUTPUT_DIR, filename)
                                    with open(filepath, "w", encoding="utf-8") as out:
                                        out.write(f"# {title}\n\n{plain}\n")

                                    extracted += 1
                                    if extracted % 100 == 0:
                                        print(f"  [{extracted + start_count:>7}] {title}")
                                        progress["count"] = extracted + start_count
                                        save_json(PROGRESS_FILE, progress)

                                    if max_articles and (extracted + start_count) >= max_articles:
                                        progress["count"] = extracted + start_count
                                        save_json(PROGRESS_FILE, progress)
                                        print(f"\nReached max ({max_articles}). Stopping.")
                                        return

                        title = None
                        text = None
                        elem.clear()

    except KeyboardInterrupt:
        pass

    progress["count"] = extracted + start_count
    save_json(PROGRESS_FILE, progress)
    print(f"\nExtracted {extracted} articles this session.")
    print(f"Total: {extracted + start_count}")
    print("Run again to continue where you left off.")


def main():
    parser = argparse.ArgumentParser(description="Download and extract Wikipedia dump")
    parser.add_argument("--max", type=int, default=0, help="Max articles to extract (0 = unlimited)")
    parser.add_argument("--reset", action="store_true", help="Clear extraction progress")
    parser.add_argument("--skip-download", action="store_true", help="Skip download (use existing dump)")
    args = parser.parse_args()

    if not args.skip_download:
        download_dump()

    if not os.path.exists(DUMP_PATH):
        print(f"Error: dump not found at {DUMP_PATH}")
        print("Run without --skip-download first.")
        sys.exit(1)

    extract_articles(max_articles=args.max, reset=args.reset)


if __name__ == "__main__":
    main()
