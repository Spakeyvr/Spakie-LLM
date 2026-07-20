"""Build broad instruction-following curricula and held-out v2 benchmarks.

The generated training examples teach transferable task families. Evaluation
uses disjoint facts, values, names, and prompt phrasings, and exact normalized
prompt overlap is rejected before any artifact is published.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path


SEED = 20260718
RAW_DIR = Path("data/chat_raw")
EVAL_DIR = Path("data/eval")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sft_row(source: str, user: str, assistant: str) -> dict:
    return {
        "source": source,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def add_unique(rows: list[dict], seen: set[str], row: dict) -> None:
    signature = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
    if signature not in seen:
        seen.add(signature)
        rows.append(row)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


TRAIN_KNOWLEDGE = [
    ("Germany", "Berlin", "its modern history, museums, and role in European politics"),
    ("Italy", "Rome", "its ancient ruins, art, and long influence on European culture"),
    ("Japan", "Tokyo", "its large economy, technology, and distinctive urban culture"),
    ("Spain", "Madrid", "its museums, royal history, and central role in Spanish government"),
    ("Canada", "Ottawa", "its national institutions and bilingual cultural heritage"),
    ("Australia", "Canberra", "its planned design and national government institutions"),
    ("Brazil", "Brasília", "its modernist architecture and purpose-built city plan"),
    ("Kenya", "Nairobi", "its regional business role and proximity to major wildlife areas"),
    ("Egypt", "Cairo", "its ancient heritage and position beside the Nile"),
    ("Norway", "Oslo", "its maritime history, museums, and surrounding fjord landscape"),
    ("Portugal", "Lisbon", "its seafaring history, hillside neighborhoods, and Atlantic setting"),
    ("Greece", "Athens", "its ancient democratic history and classical landmarks"),
    ("India", "New Delhi", "its national government and blend of historic and modern districts"),
    ("South Korea", "Seoul", "its technology sector, cultural influence, and long history"),
    ("Mexico", "Mexico City", "its scale, museums, and layers of Indigenous and colonial history"),
    ("Argentina", "Buenos Aires", "its architecture, tango tradition, and cultural life"),
    ("Thailand", "Bangkok", "its temples, markets, and importance to regional commerce"),
    ("New Zealand", "Wellington", "its harbor, film industry, and national institutions"),
    ("Austria", "Vienna", "its classical music heritage, architecture, and diplomatic role"),
    ("Ireland", "Dublin", "its literary heritage, Georgian architecture, and cultural life"),
]


def build_compound_instruction_rows() -> list[dict]:
    source = "compound_instruction_following_v2"
    rows: list[dict] = []
    seen: set[str] = set()
    knowledge_templates = [
        "What is the capital of {country}, and give one reason the city is notable?",
        "Name {country}'s capital and briefly explain what makes it significant.",
        "Answer both parts: which city is the capital of {country}, and what is one thing it is known for?",
        "Tell me the capital of {country}. Then add one short sentence about why people know the city.",
        "Give {country}'s capital first, followed by one concise reason it matters.",
    ]
    for country, capital, reason in TRAIN_KNOWLEDGE:
        answer = f"{capital} is the capital of {country}. It is notable for {reason}."
        for template in knowledge_templates:
            add_unique(rows, seen, sft_row(source, template.format(country=country), answer))

    rng = random.Random(SEED + 1)
    word_pool = [
        "amber", "birch", "cedar", "delta", "ember", "falcon", "garden", "harbor",
        "island", "jasmine", "kettle", "lantern", "meadow", "nectar", "orchid",
        "pebble", "quartz", "river", "silver", "timber", "umber", "violet",
        "willow", "xenon", "yellow", "zephyr", "acorn", "breeze", "cobalt", "daisy",
    ]
    for index in range(700):
        words = rng.sample(word_pool, rng.choice((4, 5, 6)))
        ordered = sorted(words)
        prompt = (
            f"Alphabetize these words: {', '.join(words)}. Then put their count on a second line "
            "using exactly `Count: N`."
        )
        answer = f"{', '.join(ordered)}\nCount: {len(words)}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for index in range(700):
        values = rng.sample(range(2, 80), rng.choice((5, 6, 7)))
        evens = [value for value in values if value % 2 == 0]
        prompt = (
            f"From {values}, list the even numbers in their original order. On the next line give "
            "their sum. Use exactly `Evens: ...` and `Sum: ...`."
        )
        answer = f"Evens: {', '.join(map(str, evens)) if evens else 'none'}\nSum: {sum(evens)}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    colors = ["red", "blue", "green", "gold", "silver", "purple", "orange", "white"]
    objects = ["kite", "mug", "book", "lamp", "scarf", "bowl", "chair", "clock"]
    for index in range(500):
        color, obj = rng.choice(colors), rng.choice(objects)
        number = rng.randint(10, 999)
        prompt = (
            f"Do both transformations. Write `{color} {obj}` in uppercase on the first line. "
            f"Write the digits of {number} in reverse order on the second line. Add nothing else."
        )
        answer = f"{color.upper()} {obj.upper()}\n{str(number)[::-1]}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    rng.shuffle(rows)
    return rows


def build_math_reasoning_rows() -> list[dict]:
    source = "math_reasoning_curriculum_v2"
    rows: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(SEED + 2)

    for _ in range(1200):
        a, b, c = rng.randint(2, 60), rng.randint(2, 15), rng.randint(2, 12)
        d, e = rng.randint(20, 120), rng.randint(2, 19)
        first, second = a + b * c, d - e
        prompt = (
            f"Calculate both expressions: {a} + {b} × {c}, and {d} - {e}. "
            "Output exactly two lines labeled `First:` and `Second:`."
        )
        answer = f"First: {first}\nSecond: {second}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(500):
        price = rng.randrange(20, 301, 5)
        percent = rng.choice((10, 20, 25, 30, 40, 50))
        if price * percent % 100:
            continue
        discount = price * percent // 100
        sale = price - discount
        prompt = (
            f"An item costs ${price}. It is discounted by {percent}%. Give the discount amount and "
            "the final price using `Discount: $N | Final: $N`."
        )
        answer = f"Discount: ${discount} | Final: ${sale}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(500):
        groups, per_group = rng.randint(3, 12), rng.randint(4, 18)
        removed, people = rng.randint(1, 15), rng.randint(2, 9)
        remaining = groups * per_group - removed
        if remaining <= 0:
            continue
        each, remainder = divmod(remaining, people)
        prompt = (
            f"There are {groups} trays with {per_group} tokens each. {removed} tokens are removed, "
            f"then the rest are shared equally by {people} people. How many does each get and how "
            "many remain? Use `Each: N | Remainder: N`."
        )
        answer = f"Each: {each} | Remainder: {remainder}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(500):
        hour, minute = rng.randint(0, 23), rng.choice((0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55))
        duration = rng.randint(15, 210)
        total = hour * 60 + minute + duration
        end_hour, end_minute = divmod(total % (24 * 60), 60)
        prompt = f"A session starts at {hour:02d}:{minute:02d} and lasts {duration} minutes. When does it end? Answer HH:MM only."
        answer = f"{end_hour:02d}:{end_minute:02d}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    nouns = [("mips", "larns", "blue"), ("toves", "rinns", "square"), ("zells", "pards", "warm"), ("fens", "daxes", "metal")]
    for index in range(500):
        subset, superset, forbidden = rng.choice(nouns)
        prompt = (
            f"All {subset} are {superset}. No {superset} are {forbidden}. Can any {subset} be {forbidden}? "
            "Answer Yes or No, then justify in one sentence."
        )
        answer = f"No. All {subset} are {superset}, and no {superset} are {forbidden}, so no {subset} can be {forbidden}."
        add_unique(rows, seen, sft_row(source, prompt, answer))

    people = ["Ava", "Ben", "Cora", "Drew", "Eli", "Faye", "Gus", "Hana", "Ivan", "Jade"]
    directions = [
        ("north", "east", "northeast"), ("north", "west", "northwest"),
        ("south", "east", "southeast"), ("south", "west", "southwest"),
    ]
    for _ in range(500):
        a, b, c = rng.sample(people, 3)
        first, second, result = rng.choice(directions)
        prompt = f"{a} is {first} of {b}. {b} is {second} of {c}. Where is {a} relative to {c}? Answer with one direction."
        add_unique(rows, seen, sft_row(source, prompt, result))

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for _ in range(500):
        start_index = rng.randrange(7)
        start_day = rng.randint(1, 12)
        delta = rng.randint(3, 24)
        prompt = (
            f"If August {start_day} is a {weekdays[start_index]}, what weekday is August {start_day + delta}? "
            "Answer with the weekday only."
        )
        answer = weekdays[(start_index + delta) % 7]
        add_unique(rows, seen, sft_row(source, prompt, answer))

    rng.shuffle(rows)
    return rows


def build_structured_coding_rows() -> list[dict]:
    source = "structured_coding_curriculum_v2"
    rows: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(SEED + 3)
    names = ["Amina", "Boris", "Chloe", "Diego", "Elena", "Farah", "Gavin", "Hiro", "Imani", "Jonas"]
    cities = ["Riga", "Tallinn", "Prague", "Brno", "Lyon", "Ghent", "Basel", "Turin", "Split", "Graz"]

    for _ in range(600):
        name, city, score = rng.choice(names), rng.choice(cities), rng.randint(1, 100)
        prompt = f"Return only JSON for {name} from {city} with score {score}. Use keys name, city, and score."
        answer = json.dumps({"name": name, "city": city, "score": score}, separators=(",", ":"))
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(350):
        first, second = rng.sample(names, 2)
        age1, age2 = rng.randint(18, 70), rng.randint(18, 70)
        prompt = (
            f"Output only CSV with header name,age and rows for {first}, age {age1}, and {second}, age {age2}, in that order."
        )
        answer = f"name,age\n{first},{age1}\n{second},{age2}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(250):
        items = rng.sample(["apple", "pear", "plum", "mango", "kiwi", "peach", "lime", "fig"], 3)
        colors = rng.sample(["red", "green", "yellow", "purple", "orange", "brown"], 3)
        prompt = (
            f"Output only a Markdown table with headers Item and Color and these rows: "
            + "; ".join(f"{item}/{color}" for item, color in zip(items, colors))
            + "."
        )
        answer = "| Item | Color |\n|---|---|\n" + "\n".join(
            f"| {item} | {color} |" for item, color in zip(items, colors)
        )
        add_unique(rows, seen, sft_row(source, prompt, answer))

    operations = [
        ("double", "n", "n * 2"), ("triple", "n", "n * 3"),
        ("add_five", "n", "n + 5"), ("is_even", "n", "n % 2 == 0"),
        ("is_positive", "n", "n > 0"), ("last_item", "items", "items[-1]"),
        ("text_length", "text", "len(text)"), ("first_character", "text", "text[0]"),
    ]
    for index in range(700):
        base_name, arg, expr = rng.choice(operations)
        name = f"{base_name}_{index}"
        prompt = f"Write a Python function `{name}({arg})` that returns `{expr}`. Output only code."
        answer = f"def {name}({arg}):\n    return {expr}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    for _ in range(600):
        multiplier, stop, offset = rng.randint(2, 7), rng.randint(3, 8), rng.randint(0, 5)
        expression = f"[x * {multiplier} + {offset} for x in range({stop})]"
        answer = repr([x * multiplier + offset for x in range(stop)])
        prompt = f"What does this Python expression evaluate to? `{expression}` Output only the list."
        add_unique(rows, seen, sft_row(source, prompt, answer))

    rng.shuffle(rows)
    return rows


def build_grounded_multitask_rows() -> list[dict]:
    source = "grounded_multitask_curriculum_v2"
    rows: list[dict] = []
    seen: set[str] = set()
    rng = random.Random(SEED + 4)
    names = ["Arun", "Bella", "Cleo", "Dara", "Emil", "Freya", "Gita", "Hugo", "Ines", "Jamal", "Kira", "Luca"]
    objects = ["red badges", "blue cards", "silver tokens", "paper maps", "green flags", "wooden blocks"]

    for _ in range(1400):
        owner, helper = rng.sample(names, 2)
        obj = rng.choice(objects)
        start, given = rng.randint(8, 60), rng.randint(1, 7)
        if given >= start:
            continue
        passage = (
            f"{owner} manages the supply cabinet. It contains {start} {obj}. "
            f"On Tuesday, {helper} takes {given} of them for a workshop. Nobody else uses the cabinet that day."
        )
        prompt = (
            f"Passage: {passage}\n\nAnswer both questions using exactly two lines: "
            "Who manages the cabinet? How many items remain? Use `Manager:` and `Remaining:`."
        )
        answer = f"Manager: {owner}\nRemaining: {start - given}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    activities = ["painting class", "coding club", "music practice", "book group", "science lab"]
    rooms = ["Room A", "Room B", "Hall 2", "Studio 4", "Lab 3"]
    for _ in range(800):
        activity, weekday, room = rng.choice(activities), rng.choice(weekdays), rng.choice(rooms)
        other_day = rng.choice([day for day in weekdays if day != weekday])
        passage = (
            f"The {activity} meets in {room} on {weekday}. It does not meet on {other_day}. "
            "Participants should arrive ten minutes early."
        )
        prompt = (
            f"Read this note: {passage}\nGive the meeting day and room. Output exactly `Day: ... | Room: ...`."
        )
        answer = f"Day: {weekday} | Room: {room}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    replacements = [
        ("Send the notes now.", "Could you please send the notes when you have a moment?"),
        ("Fix this today.", "Could you please fix this today?"),
        ("Give me the schedule.", "Could you please share the schedule?"),
        ("Check the numbers again.", "Could you please check the numbers again?"),
    ]
    for index in range(700):
        original, polite = rng.choice(replacements)
        tag = rng.choice(("READY", "REVIEWED", "UPDATED", "COMPLETE"))
        prompt = (
            f"Rewrite `{original}` as a polite professional request. On a second line write `Status: {tag}` exactly."
        )
        answer = f"{polite}\nStatus: {tag}"
        add_unique(rows, seen, sft_row(source, prompt, answer))

    rng.shuffle(rows)
    return rows


EVAL_KNOWLEDGE = [
    ("France", "Paris", ["culture", "art", "history", "landmark", "museum", "architecture", "fashion"]),
    ("Peru", "Lima", ["coast", "history", "food", "culture", "colonial"]),
    ("Iceland", "Reykjavík", ["geothermal", "culture", "northern", "harbor", "government"]),
    ("Vietnam", "Hanoi", ["history", "culture", "architecture", "government"]),
    ("Morocco", "Rabat", ["history", "architecture", "government", "culture"]),
    ("Finland", "Helsinki", ["design", "architecture", "Baltic", "culture"]),
    ("Chile", "Santiago", ["Andes", "culture", "economy", "mountain"]),
    ("Croatia", "Zagreb", ["culture", "architecture", "history", "museum"]),
    ("Senegal", "Dakar", ["Atlantic", "culture", "music", "port"]),
    ("Nepal", "Kathmandu", ["Himalaya", "temple", "culture", "history"]),
    ("Uruguay", "Montevideo", ["coast", "culture", "port", "architecture"]),
    ("Jordan", "Amman", ["history", "hills", "culture", "ancient"]),
]


def exact(value: str | list) -> dict:
    return {"type": "exact", "value": [value] if isinstance(value, str) else value}


def benchmark_rows(split: str) -> list[dict]:
    if split not in {"core", "fresh"}:
        raise ValueError(split)
    per_category = 8 if split == "core" else 4
    offset = 0 if split == "core" else 8
    prefix = "if2-core" if split == "core" else "if2-fresh"
    rows: list[dict] = []
    rng = random.Random(SEED + (10 if split == "core" else 11))

    for index, (country, capital, keywords) in enumerate(EVAL_KNOWLEDGE[offset:offset + per_category], start=1):
        prompt = (
            "What’s the capital of France and why’s it so special?"
            if split == "core" and index == 1
            else f"Name the capital of {country} and explain briefly why that city is notable. Answer both parts."
        )
        rows.append({
            "id": f"{prefix}-compound-knowledge-{index:02d}",
            "category": "compound_knowledge",
            "prompt": prompt,
            "check": {
                "type": "all_of",
                "value": [
                    {"type": "contains_all", "value": [capital]},
                    {"type": "contains_any", "value": keywords},
                ],
            },
            "unit_checks": [
                {"name": "correct_capital", "check": {"type": "contains_all", "value": [capital]}},
                {"name": "notability_reason", "check": {"type": "contains_any", "value": keywords}},
            ],
            "semantic_requirements": {"contains_one_of": keywords, "must_answer_all_parts": True},
        })

    benchmark_words = ["opal", "maple", "canyon", "rocket", "linen", "tulip", "walnut", "coral", "piano", "saffron", "marble", "comet"]
    for index in range(per_category):
        words = rng.sample(benchmark_words, 5)
        answer = f"{', '.join(sorted(words))}\nCount: 5"
        rows.append({
            "id": f"{prefix}-compound-instruction-{index + 1:02d}",
            "category": "compound_instruction",
            "prompt": f"Put these words in alphabetical order: {', '.join(words)}. Then put `Count: 5` on a new line. Add nothing else.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "all_words", "check": {"type": "contains_all", "value": sorted(words)}},
                {"name": "count_line", "check": {"type": "contains_all", "value": ["Count: 5"]}},
                {"name": "exact_format", "check": exact(answer)},
            ],
        })

    for index in range(per_category):
        a, b, c, d, e = rng.randint(11, 70), rng.randint(3, 14), rng.randint(2, 9), rng.randint(35, 120), rng.randint(3, 22)
        answer = f"First: {a + b * c}\nSecond: {d - e}"
        rows.append({
            "id": f"{prefix}-arithmetic-{index + 1:02d}",
            "category": "arithmetic",
            "prompt": f"Compute {a} + {b} × {c} and {d} - {e}. Return exactly two lines labeled First and Second.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "first_result", "check": {"type": "contains_all", "value": ["First:", str(a + b * c)]}},
                {"name": "second_result", "check": {"type": "contains_all", "value": ["Second:", str(d - e)]}},
                {"name": "exact_format", "check": exact(answer)},
            ],
        })

    for index in range(per_category):
        groups, each, removed, people = rng.randint(4, 11), rng.randint(5, 16), rng.randint(2, 12), rng.randint(3, 8)
        remaining = groups * each - removed
        share, remainder = divmod(remaining, people)
        answer = f"Each: {share} | Remainder: {remainder}"
        rows.append({
            "id": f"{prefix}-multi-step-math-{index + 1:02d}",
            "category": "multi_step_math",
            "prompt": f"There are {groups} boxes of {each} beads. After {removed} are lost, the rest are split among {people} children. Give each child's share and the remainder using `Each: N | Remainder: N`.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "share", "check": {"type": "contains_all", "value": ["Each:", str(share)]}},
                {"name": "remainder", "check": {"type": "contains_all", "value": ["Remainder:", str(remainder)]}},
                {"name": "exact_format", "check": exact(answer)},
            ],
        })

    logic_sets = [
        ("vims", "tars", "quiet"), ("nors", "pels", "round"),
        ("zibs", "loms", "cold"), ("fars", "kens", "wooden"),
        ("beks", "rals", "silver"), ("dops", "mirs", "soft"),
        ("gans", "wels", "tall"), ("hups", "yars", "striped"),
        ("jeks", "cals", "warm"), ("kivs", "sors", "metal"),
        ("paks", "nems", "bright"), ("ruds", "fims", "square"),
    ]
    for index in range(per_category):
        a, b, adjective = logic_sets[offset + index]
        answer = f"No. All {a} are {b}, and no {b} are {adjective}, so no {a} can be {adjective}."
        rows.append({
            "id": f"{prefix}-logic-{index + 1:02d}",
            "category": "logic",
            "prompt": f"All {a} are {b}. No {b} are {adjective}. Can any {a} be {adjective}? Answer and justify in one sentence.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "correct_decision", "check": {"type": "regex", "value": r"^\s*No[.\s]"}},
                {"name": "uses_given_relation", "check": {"type": "contains_all", "value": [a, b, adjective]}},
                {"name": "complete_reasoning", "check": exact(answer)},
            ],
        })

    directions = [("north", "east", "northeast"), ("north", "west", "northwest"), ("south", "east", "southeast"), ("south", "west", "southwest")]
    eval_names = ["Mara", "Niko", "Oren", "Pia", "Quin", "Rosa", "Sami", "Tess", "Umar", "Vera", "Wade", "Yara"]
    for index in range(per_category):
        a, b, c = rng.sample(eval_names, 3)
        first, second, result = directions[index % 4]
        rows.append({
            "id": f"{prefix}-spatial-{index + 1:02d}",
            "category": "spatial",
            "prompt": f"{a} is {first} of {b}, and {b} is {second} of {c}. Where is {a} relative to {c}? Reply with one direction only.",
            "check": exact(result),
            "unit_checks": [
                {"name": "correct_direction", "check": {"type": "contains_all", "value": [result]}},
                {"name": "one_direction_only", "check": exact(result)},
            ],
        })

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for index in range(per_category):
        start = rng.randrange(7)
        day, delta = rng.randint(2, 12), rng.randint(8, 19)
        result = weekdays[(start + delta) % 7]
        rows.append({
            "id": f"{prefix}-date-time-{index + 1:02d}",
            "category": "date_time",
            "prompt": f"If September {day} is a {weekdays[start]}, what weekday is September {day + delta}? Answer with the weekday only.",
            "check": exact(result),
            "unit_checks": [
                {"name": "correct_weekday", "check": {"type": "contains_all", "value": [result]}},
                {"name": "weekday_only", "check": exact(result)},
            ],
        })

    code_ops = [("quadruple", "n", "n * 4"), ("subtract_two", "n", "n - 2"), ("is_odd", "n", "n % 2 == 1"), ("list_size", "items", "len(items)")]
    for index in range(per_category):
        base, arg, expr = code_ops[index % len(code_ops)]
        name = f"{base}_test_{offset + index}"
        answer = f"def {name}({arg}):\n    return {expr}"
        rows.append({
            "id": f"{prefix}-coding-{index + 1:02d}",
            "category": "coding",
            "prompt": f"Write a Python function `{name}({arg})` that returns `{expr}`. Output only code.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "function_signature", "check": {"type": "contains_all", "value": [f"def {name}({arg}):"]}},
                {"name": "return_expression", "check": {"type": "contains_all", "value": [f"return {expr}"]}},
                {"name": "code_only", "check": exact(answer)},
            ],
        })

    for index in range(per_category):
        name = eval_names[(offset + index) % len(eval_names)]
        city = ["Delft", "Porto", "Bern", "Lodz", "Nantes", "Pisa", "Leeds", "Uppsala", "Oulu", "Siena", "Linz", "Cork"][(offset + index) % 12]
        score = 70 + offset + index
        value = {"name": name, "city": city, "score": score}
        rows.append({
            "id": f"{prefix}-structured-{index + 1:02d}",
            "category": "structured_output",
            "prompt": f"Return only valid JSON for {name} from {city} with score {score}. Use keys name, city, and score.",
            "check": {"type": "exact_json", "value": value},
            "unit_checks": [
                {"name": "name_field", "check": {"type": "json", "required_keys": ["name"], "expected": {"name": name}}},
                {"name": "city_field", "check": {"type": "json", "required_keys": ["city"], "expected": {"city": city}}},
                {"name": "score_field", "check": {"type": "json", "required_keys": ["score"], "expected": {"score": score}}},
                {"name": "exact_json", "check": {"type": "exact_json", "value": value}},
            ],
        })

    for index in range(per_category):
        owner, helper = rng.sample(eval_names, 2)
        start, used = rng.randint(20, 55), rng.randint(3, 11)
        passage = f"{owner} runs the archive. It holds {start} folders. {helper} checks out {used} folders, and nobody else takes any."
        answer = f"Manager: {owner}\nRemaining: {start - used}"
        rows.append({
            "id": f"{prefix}-grounded-{index + 1:02d}",
            "category": "grounded_multi_part_qa",
            "prompt": f"Passage: {passage}\n\nWho runs the archive and how many folders remain? Use exactly two lines labeled Manager and Remaining.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "manager", "check": {"type": "contains_all", "value": ["Manager:", owner]}},
                {"name": "remaining", "check": {"type": "contains_all", "value": ["Remaining:", str(start - used)]}},
                {"name": "exact_format", "check": exact(answer)},
            ],
        })

    polite_pairs = [
        ("Send the draft tonight.", "Could you please send the draft tonight?"),
        ("Review these figures.", "Could you please review these figures?"),
        ("Update the calendar.", "Could you please update the calendar?"),
        ("Bring the forms tomorrow.", "Could you please bring the forms tomorrow?"),
        ("Share the agenda today.", "Could you please share the agenda today?"),
        ("Check the invoice again.", "Could you please check the invoice again?"),
        ("Confirm the room booking.", "Could you please confirm the room booking?"),
        ("Email the summary.", "Could you please email the summary?"),
        ("Print the final chart.", "Could you please print the final chart?"),
        ("Organize the receipts.", "Could you please organize the receipts?"),
        ("Call the supplier.", "Could you please call the supplier?"),
        ("Save the revised file.", "Could you please save the revised file?"),
    ]
    for index in range(per_category):
        original, polite = polite_pairs[offset + index]
        status = ["PENDING", "READY", "CHECKED", "SENT"][(offset + index) % 4]
        answer = f"{polite}\nStatus: {status}"
        rows.append({
            "id": f"{prefix}-transform-{index + 1:02d}",
            "category": "text_transformation",
            "prompt": f"Make `{original}` polite and professional. Then write `Status: {status}` on a second line. Add nothing else.",
            "check": exact(answer),
            "unit_checks": [
                {"name": "polite_rewrite", "check": {"type": "contains_all", "value": ["please"]}},
                {"name": "status_line", "check": {"type": "contains_all", "value": [f"Status: {status}"]}},
                {"name": "exact_format", "check": exact(answer)},
            ],
        })

    topics = [
        ("public libraries", ["books", "access", "community"]),
        ("daily exercise", ["health", "strength", "mood"]),
        ("recycling", ["waste", "materials", "resources"]),
        ("planning ahead", ["time", "steps", "confusion"]),
        ("urban trees", ["shade", "air", "wildlife"]),
        ("reading regularly", ["knowledge", "vocabulary", "focus"]),
        ("teamwork", ["ideas", "responsibility", "support"]),
        ("public transport", ["traffic", "cost", "emissions"]),
        ("sleep", ["energy", "memory", "health"]),
        ("meal planning", ["waste", "money", "nutrition"]),
        ("learning music", ["practice", "creativity", "coordination"]),
        ("community gardens", ["food", "neighbors", "green space"]),
    ]
    for index in range(per_category):
        topic, keywords = topics[offset + index]
        rows.append({
            "id": f"{prefix}-explanation-{index + 1:02d}",
            "category": "explanation_planning",
            "prompt": f"Give two distinct benefits of {topic}. Use exactly two bullet points, one benefit per bullet.",
            "check": {
                "type": "all_of",
                "value": [
                    {"type": "contains_all", "value": keywords[:2]},
                    {"type": "regex", "value": r"^\s*[-*]\s+.+\n\s*[-*]\s+.+\s*$"},
                ],
            },
            "unit_checks": [
                {"name": "first_benefit", "check": {"type": "contains_all", "value": [keywords[0]]}},
                {"name": "second_benefit", "check": {"type": "contains_all", "value": [keywords[1]]}},
                {"name": "two_bullets", "check": {"type": "regex", "value": r"^\s*[-*]\s+.+\n\s*[-*]\s+.+\s*$"}},
            ],
            "semantic_requirements": {"exact_bullet_count": 2, "distinct_benefits": True},
        })

    if len(rows) != per_category * 12:
        raise AssertionError(f"unexpected {split} benchmark size: {len(rows)}")
    return rows


def validate_artifacts(training: dict[Path, list[dict]], eval_sets: dict[Path, list[dict]]) -> dict:
    training_prompts = {
        normalized(message["content"])
        for rows in training.values()
        for row in rows
        for message in row["messages"]
        if message["role"] == "user"
    }
    eval_prompts: set[str] = set()
    for path, rows in eval_sets.items():
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate benchmark ids in {path}")
        for row in rows:
            prompt = normalized(row["prompt"])
            if prompt in training_prompts:
                raise ValueError(f"exact train/eval prompt overlap: {row['id']}")
            if prompt in eval_prompts:
                raise ValueError(f"duplicate prompt across eval sets: {row['id']}")
            eval_prompts.add(prompt)
    return {
        "training_rows": {str(path): len(rows) for path, rows in training.items()},
        "evaluation_rows": {str(path): len(rows) for path, rows in eval_sets.items()},
        "exact_train_eval_prompt_overlap": 0,
        "seed": SEED,
    }


def main() -> int:
    training = {
        RAW_DIR / "compound_instruction_following_v2.jsonl": build_compound_instruction_rows(),
        RAW_DIR / "math_reasoning_curriculum_v2.jsonl": build_math_reasoning_rows(),
        RAW_DIR / "structured_coding_curriculum_v2.jsonl": build_structured_coding_rows(),
        RAW_DIR / "grounded_multitask_curriculum_v2.jsonl": build_grounded_multitask_rows(),
    }
    eval_sets = {
        EVAL_DIR / "instruction_reasoning_core_v2.jsonl": benchmark_rows("core"),
        EVAL_DIR / "instruction_reasoning_fresh_v2.jsonl": benchmark_rows("fresh"),
    }
    report = validate_artifacts(training, eval_sets)
    for path, rows in {**training, **eval_sets}.items():
        write_jsonl(path, rows)
        print(f"{len(rows):,} rows -> {path}")
    report["sha256"] = {str(path): sha256(path) for path in {**training, **eval_sets}}
    manifest = EVAL_DIR / "instruction_reasoning_v2_manifest.json"
    temp = manifest.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temp.replace(manifest)
    print(f"Manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; no partial curriculum or benchmark is considered valid.")
        raise SystemExit(130)
