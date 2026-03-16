"""Download and convert public chat datasets to Spakie JSONL format.

Downloads the Stanford Alpaca dataset (52k instruction-response pairs)
and converts it to data/chat/train.jsonl.

Usage:
    python scripts/download_sft_data.py
    python scripts/download_sft_data.py --max 5000    # limit to 5k examples
"""

import argparse
import json
import os
import sys

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
HEADERS = {"User-Agent": "SpakieLLM/1.0 (educational language model project)"}

SYSTEM_MSG = "You are Spakie, a helpful assistant."


def download_alpaca() -> list[dict]:
    """Download the Alpaca 52k dataset."""
    print("Downloading Alpaca dataset (52k examples)...")
    resp = requests.get(ALPACA_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    print(f"Downloaded {len(data)} examples.")
    return data


def convert_to_spakie_format(examples: list[dict]) -> list[dict]:
    """Convert Alpaca format to Spakie chat JSONL format."""
    converted = []
    for ex in examples:
        instruction = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        output = ex.get("output", "").strip()

        if not instruction or not output:
            continue

        # Combine instruction + input if input exists
        if inp:
            user_msg = f"{instruction}\n\n{inp}"
        else:
            user_msg = instruction

        converted.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output},
            ]
        })

    return converted


def main():
    parser = argparse.ArgumentParser(description="Download SFT training data")
    parser.add_argument("--max", type=int, default=0, help="Max examples (0 = all)")
    args = parser.parse_args()

    config = SpakieConfig()
    os.makedirs(config.chat_data_dir, exist_ok=True)

    raw = download_alpaca()
    converted = convert_to_spakie_format(raw)

    if args.max > 0:
        converted = converted[:args.max]

    out_path = os.path.join(config.chat_data_dir, "train.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in converted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(converted)} examples to {out_path}")


if __name__ == "__main__":
    main()
