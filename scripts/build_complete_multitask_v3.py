"""Build a completion-focused SFT curriculum without benchmark prompt leakage.

The examples emphasize finishing every requested component across knowledge,
grounded extraction, calculations, transformations, explanations, and JSON.
Outputs are deterministic and published atomically so Ctrl+C cannot leave a
partially written dataset in place.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


SEED = 20260718 + 30
OUTPUT = Path("data/chat_raw/complete_multi_part_answers_v3.jsonl")
MANIFEST = OUTPUT.with_suffix(OUTPUT.suffix + ".manifest.json")
EVAL_PATHS = (
    Path("data/eval/instruction_reasoning_core_v2.jsonl"),
    Path("data/eval/instruction_reasoning_fresh_v2.jsonl"),
)


KNOWLEDGE = (
    ("Germany", "Berlin", "its museums, modern history, and role in European politics"),
    ("Italy", "Rome", "its ancient ruins, art, and long cultural influence"),
    ("Japan", "Tokyo", "its technology, large economy, and distinctive urban culture"),
    ("Spain", "Madrid", "its major museums, royal history, and national institutions"),
    ("Canada", "Ottawa", "its national institutions and bilingual heritage"),
    ("Australia", "Canberra", "its planned design and national government institutions"),
    ("Brazil", "Brasilia", "its modernist architecture and purpose-built city plan"),
    ("Kenya", "Nairobi", "its regional business role and nearby wildlife areas"),
    ("Egypt", "Cairo", "its ancient heritage and location beside the Nile"),
    ("Norway", "Oslo", "its maritime history, museums, and fjord setting"),
    ("Portugal", "Lisbon", "its seafaring history and Atlantic hillside neighborhoods"),
    ("Greece", "Athens", "its classical landmarks and ancient democratic history"),
    ("India", "New Delhi", "its national government and historic monuments"),
    ("South Korea", "Seoul", "its technology sector, cultural influence, and long history"),
    ("Mexico", "Mexico City", "its museums and layers of Indigenous and colonial history"),
    ("Argentina", "Buenos Aires", "its architecture, tango tradition, and cultural life"),
    ("Thailand", "Bangkok", "its temples, markets, and regional commercial importance"),
    ("New Zealand", "Wellington", "its harbor, film industry, and national institutions"),
    ("Austria", "Vienna", "its classical music heritage, architecture, and diplomatic role"),
    ("Ireland", "Dublin", "its literary heritage and Georgian architecture"),
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def row(source: str, prompt: str, answer: str) -> dict:
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def add(rows: list[dict], seen: set[str], prompt: str, answer: str) -> None:
    signature = normalize(prompt)
    if signature in seen:
        return
    seen.add(signature)
    rows.append(row("complete_multi_part_answers_v3", prompt, answer))


def build_rows() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    seen: set[str] = set()

    # Conversational and explicit wordings teach that a second clause is not
    # optional. Held-out benchmark countries are deliberately absent.
    openers = (
        "Quick question {n}: what's the capital of {country}, and why is that city well known?",
        "For item {n}, tell me {country}'s capital and add one reason it matters.",
        "Answer both parts for case {n}: name the capital of {country}, then say what makes it notable.",
        "I need two details for entry {n}. Which city is {country}'s capital, and what is it known for?",
        "Capital check {n}: identify {country}'s capital and briefly explain its significance.",
        "Give a complete reply for card {n}: the capital of {country}, plus one notable feature.",
    )
    for n in range(1, 1201):
        country, capital, reason = KNOWLEDGE[(n - 1) % len(KNOWLEDGE)]
        prompt = openers[(n - 1) % len(openers)].format(n=n, country=country)
        answer = f"{capital} is the capital of {country}. It is well known for {reason}."
        add(rows, seen, prompt, answer)

    names = ("Ada", "Bruno", "Cleo", "Dara", "Enzo", "Fara", "Gita", "Hugo", "Iris", "Juno")
    objects = ("crates", "folders", "tickets", "samples", "notebooks", "parcels")
    locations = ("depot", "cabinet", "office", "workroom", "studio", "warehouse")
    for n in range(1200):
        owner, helper = rng.sample(names, 2)
        obj, location = rng.choice(objects), rng.choice(locations)
        start = rng.randint(20, 99)
        removed = rng.randint(2, min(18, start - 1))
        prompt = (
            f"Record {n + 1}: {owner} manages the {location}, which starts with {start} {obj}. "
            f"{helper} removes {removed}, and nobody else removes any. Who manages it and how many "
            "remain? Reply on exactly two lines labeled `Manager:` and `Remaining:`."
        )
        add(rows, seen, prompt, f"Manager: {owner}\nRemaining: {start - removed}")

    verbs = ("send", "review", "update", "bring", "share", "check", "confirm", "email", "print", "organize")
    nouns = ("draft", "figures", "calendar", "forms", "agenda", "invoice", "booking", "summary", "chart", "receipts")
    statuses = ("PENDING", "READY", "CHECKED", "SENT")
    for n in range(800):
        verb = verbs[n % len(verbs)]
        noun = nouns[(n * 3) % len(nouns)]
        status = statuses[(n * 5) % len(statuses)]
        prompt = (
            f"Task {n + 1}: rewrite `{verb.capitalize()} the {noun}.` as a polite request. "
            f"On the second line write exactly `Status: {status}`. Include both lines and nothing else."
        )
        answer = f"Could you please {verb} the {noun}?\nStatus: {status}"
        add(rows, seen, prompt, answer)

    topics = (
        ("walking regularly", "health", "mood"),
        ("keeping a budget", "spending", "saving"),
        ("reading books", "knowledge", "vocabulary"),
        ("planting trees", "shade", "air quality"),
        ("planning meals", "food waste", "money"),
        ("working in teams", "ideas", "shared responsibility"),
        ("using a bicycle", "exercise", "lower emissions"),
        ("learning a language", "communication", "cultural understanding"),
    )
    for n in range(600):
        topic, first, second = topics[n % len(topics)]
        prompt = (
            f"Response {n + 1}: give two different benefits of {topic}. Use exactly two bullet "
            "points and make sure both requested benefits are present."
        )
        answer = f"- It can improve {first}.\n- It can support {second}."
        add(rows, seen, prompt, answer)

    for n in range(800):
        name = f"Person{n + 101}"
        city = f"Town{(n * 17) % 997 + 101}"
        score = (n * 13) % 100 + 1
        prompt = (
            f"JSON record {n + 1}: return only valid JSON for name {name}, city {city}, and score "
            f"{score}. Use exactly the keys name, city, and score."
        )
        answer = json.dumps({"name": name, "city": city, "score": score}, separators=(",", ":"))
        add(rows, seen, prompt, answer)

    for n in range(800):
        a, b, c = rng.randint(8, 75), rng.randint(2, 12), rng.randint(2, 9)
        d, e = rng.randint(30, 140), rng.randint(2, 24)
        prompt = (
            f"Calculation card {n + 1}: solve {a} + {b} x {c} and {d} - {e}. Give both answers "
            "on exactly two lines labeled `First:` and `Second:`."
        )
        add(rows, seen, prompt, f"First: {a + b * c}\nSecond: {d - e}")

    rng.shuffle(rows)
    return rows


def eval_prompts() -> set[str]:
    prompts: set[str] = set()
    for path in EVAL_PATHS:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                prompts.add(normalize(json.loads(line)["prompt"]))
    return prompts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    rows = build_rows()
    prompts = [normalize(item["messages"][0]["content"]) for item in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate normalized training prompts")
    overlap = set(prompts) & eval_prompts()
    if overlap:
        raise ValueError(f"exact evaluation overlap: {sorted(overlap)[:3]}")

    atomic_write(OUTPUT, "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
    manifest = {
        "path": str(OUTPUT),
        "rows": len(rows),
        "seed": SEED,
        "sha256": sha256(OUTPUT),
        "exact_eval_prompt_overlap": 0,
        "purpose": "complete all requested components across transferable task families",
        "excluded_eval_entities": [
            "France", "Peru", "Iceland", "Vietnam", "Morocco", "Finland",
            "Chile", "Croatia", "Senegal", "Nepal", "Uruguay", "Jordan",
        ],
    }
    atomic_write(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows to {OUTPUT}")
    print(f"SHA-256: {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no partial artifact was published.")
        raise SystemExit(130)
