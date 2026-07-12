"""Generate varied, deterministic SFT curricula for broad weak-skill classes."""

from __future__ import annotations

import json
import os
import random
import sys


OUTPUT_DIR = "data/raw_chat"


def example(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def write_jsonl(name: str, rows: list[dict]) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"{len(rows):4} {path}")


def expand_prompt_phrasings(rows: list[dict]) -> list[dict]:
    """Add instruction phrasings without changing the underlying task."""
    expanded: list[dict] = []
    for row in rows:
        user = row["messages"][0]["content"]
        assistant = row["messages"][1]["content"]
        prompts = (
            user,
            f"Task: {user}",
            f"Please answer carefully. {user}",
            f"Solve this request accurately: {user}",
            f"Follow the requested format exactly. {user}",
            f"Provide the direct answer. {user}",
        )
        expanded.extend(example(prompt, assistant) for prompt in prompts)
    return expanded


def arithmetic_rows() -> list[dict]:
    rows: list[dict] = []
    for price in (40, 60, 90, 140, 160, 220, 360, 480):
        for percent in (10, 20, 25, 30, 40):
            if price * percent % 100:
                continue
            discount = price * percent // 100
            final = price - discount
            rows.append(example(
                f"An item costs ${price} and is discounted by {percent}%. What is the sale price? Show the calculation briefly.",
                f"The discount is {percent}% of ${price} = ${discount}. The sale price is ${price} - ${discount} = ${final}."
            ))
    fractions = [(1, 2, 1, 3), (3, 4, 1, 6), (5, 8, 1, 4), (7, 10, 2, 5), (2, 3, 1, 5)]
    from fractions import Fraction
    for a, b, c, d in fractions:
        for op in ("plus", "minus"):
            result = Fraction(a, b) + Fraction(c, d) if op == "plus" else Fraction(a, b) - Fraction(c, d)
            symbol = "+" if op == "plus" else "-"
            rows.append(example(
                f"What is {a}/{b} {op} {c}/{d}? Give the simplified fraction.",
                f"Using a common denominator, {a}/{b} {symbol} {c}/{d} = {result.numerator}/{result.denominator}."
            ))
    for start_h in (7, 9, 12, 15, 18, 21):
        for start_m, duration in ((10, 85), (25, 130), (40, 55), (50, 145)):
            total = start_h * 60 + start_m + duration
            end_h, end_m = (total // 60) % 24, total % 60
            rows.append(example(
                f"An event starts at {start_h:02d}:{start_m:02d} and lasts {duration} minutes. When does it end?",
                f"Adding {duration} minutes gives {end_h:02d}:{end_m:02d}."
            ))
    for groups, per_group, removed, people in ((4, 8, 7, 5), (6, 9, 12, 7), (5, 12, 10, 5), (8, 7, 11, 5), (3, 15, 9, 6)):
        remaining = groups * per_group - removed
        each, rem = divmod(remaining, people)
        rows.append(example(
            f"There are {groups} boxes with {per_group} counters each. After {removed} are removed, the rest are shared among {people} people. How many does each person get and how many remain?",
            f"There are {groups} × {per_group} = {groups * per_group} counters, then {remaining} remain. Each person gets {each}, with {rem} remaining."
        ))
    return rows


def structured_rows() -> list[dict]:
    rows: list[dict] = []
    people = [("Amina", 24), ("Leo", 31), ("Priya", 28), ("Mateo", 36), ("Nora", 42), ("Sam", 19)]
    for name, age in people:
        rows.append(example(
            f"Return only valid JSON with keys name and age for {name}, age {age}.",
            json.dumps({"name": name, "age": age}, separators=(",", ":"))
        ))
    cities = [("Rome", "Italy"), ("Kyoto", "Japan"), ("Lima", "Peru"), ("Accra", "Ghana"), ("Oslo", "Norway")]
    for city, country in cities:
        rows.append(example(
            f"Return only valid JSON with keys city and country for {city} in {country}.",
            json.dumps({"city": city, "country": country}, separators=(",", ":"))
        ))
    descriptors = [("gentle rain", 2), ("bright meadow", 3), ("quiet library", 4), ("cold morning", 3)]
    answers = {2: "Soft steady", 3: "Fresh bright peaceful", 4: "Shelves hold quiet knowledge"}
    for subject, count in descriptors:
        rows.append(example(
            f"Reply with exactly {count} words describing a {subject}. Do not add punctuation.", answers[count]
        ))
    for number, truth in ((27, "YES"), (35, "YES"), (41, "NO"), (57, "YES"), (62, "NO")):
        rows.append(example(
            f"Answer only YES or NO: Is {number} divisible by 3?", truth
        ))
    tables = [
        ("Tool", "Use", [("hammer", "driving nails"), ("saw", "cutting wood")]),
        ("Bird", "Color", [("raven", "black"), ("swan", "white")]),
        ("Country", "Capital", [("Spain", "Madrid"), ("Kenya", "Nairobi")]),
    ]
    for h1, h2, values in tables:
        body = f"| {h1} | {h2} |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in values)
        rows.append(example(
            f"Output only a Markdown table with headers {h1} and {h2}, containing {values[0][0]}/{values[0][1]} and {values[1][0]}/{values[1][1]}.", body
        ))
    formats = [("public transit", "cheap travel", "fixed schedules"), ("gardening", "fresh produce", "regular upkeep"), ("online study", "flexible access", "screen fatigue")]
    for topic, benefit, drawback in formats:
        rows.append(example(
            f"Give one benefit and one drawback of {topic}. Use exactly: Benefit: ... | Drawback: ...",
            f"Benefit: {benefit} | Drawback: {drawback}"
        ))
    return rows


def logic_rows() -> list[dict]:
    rows: list[dict] = []
    for month, start_day, delta, result in (
        ("June", "Monday", 9, "Wednesday"), ("July", "Thursday", 12, "Tuesday"),
        ("September", "Sunday", 15, "Monday"), ("November", "Tuesday", 10, "Friday"),
    ):
        rows.append(example(
            f"If {month} 1 is a {start_day}, what weekday is {month} {1 + delta}?",
            f"It is {delta} days later. Since {delta} modulo 7 is {delta % 7}, the weekday is {result}."
        ))
    spatial = [
        ("Iris", "north", "Milo", "east", "Tara", "northeast"),
        ("Omar", "south", "Lena", "west", "Kai", "southwest"),
        ("Pia", "north", "Ravi", "west", "Zoe", "northwest"),
        ("Uma", "south", "Theo", "east", "Bea", "southeast"),
    ]
    for a, d1, b, d2, c, result in spatial:
        rows.append(example(
            f"{a} is {d1} of {b}. {b} is {d2} of {c}. Where is {a} relative to {c}?",
            f"{a} is {result} of {c}."
        ))
    sets = [("sparrows", "birds", "mammals"), ("roses", "plants", "machines"), ("cubes", "solids", "songs"), ("salmon", "fish", "reptiles")]
    for a, b, c in sets:
        rows.append(example(
            f"All {a} are {b}. No {b} are {c}. Can any {a} be {c}?",
            f"No. All {a} are {b}, and no {b} are {c}, so no {a} can be {c}."
        ))
    ambiguous = [
        ("What is the weather there?", "Which location and date do you mean?"),
        ("When does it start?", "What event do you mean, and in which time zone?"),
        ("How much does the ticket cost?", "Which event, route, or venue and which ticket type do you mean?"),
        ("Did the Lions win?", "Which Lions team, sport, game, and date do you mean?"),
        ("What is the population of Cambridge?", "Which Cambridge do you mean, such as the city in England or one in North America?"),
    ]
    rows.extend(example(q, a) for q, a in ambiguous)
    return rows


def language_rows() -> list[dict]:
    rows: list[dict] = []
    corrections = [
        ("They is ready to leave.", "They are ready to leave."),
        ("He don't enjoy coffee.", "He doesn't enjoy coffee."),
        ("We was late for class.", "We were late for class."),
        ("She didn't bring no keys.", "She didn't bring any keys."),
        ("The dogs runs quickly.", "The dogs run quickly."),
        ("I have went home.", "I have gone home."),
    ]
    rows.extend(example(f"Correct the grammar and output only the corrected sentence: {bad}", good) for bad, good in corrections)
    passive = [
        ("The mechanic repaired the engine.", "The engine was repaired by the mechanic."),
        ("The team completed the project.", "The project was completed by the team."),
        ("The storm damaged the roof.", "The roof was damaged by the storm."),
        ("The editor reviewed the article.", "The article was reviewed by the editor."),
    ]
    rows.extend(example(f"Rewrite in passive voice: {active}", answer) for active, answer in passive)
    summaries = [
        ("Walking regularly can strengthen the heart, improve mood, and support healthy sleep.", "Regular walking benefits heart health, mood, and sleep."),
        ("Libraries lend books and provide quiet spaces, internet access, and community programs.", "Libraries provide books, workspaces, internet access, and community services."),
        ("Plants reduce soil erosion because their roots hold soil while leaves soften the impact of rain.", "Plant roots and leaves help prevent soil erosion."),
        ("Planning meals can reduce food waste, save money, and make balanced eating easier.", "Meal planning reduces waste and cost while supporting balanced eating."),
    ]
    rows.extend(example(f"Summarize in one sentence: {text}", answer) for text, answer in summaries)
    return rows


def coding_rows() -> list[dict]:
    rows: list[dict] = []
    functions = [
        ("triple", "n", "n * 3"), ("is_negative", "n", "n < 0"),
        ("first_item", "items", "items[0]"), ("string_length", "text", "len(text)"),
        ("is_multiple_of_five", "n", "n % 5 == 0"),
    ]
    for name, arg, expr in functions:
        rows.append(example(
            f"Write a Python function named {name}({arg}) that returns the appropriate result. Output only code.",
            f"def {name}({arg}):\n    return {expr}"
        ))
    for multiplier in (2, 3, 4, 5):
        for stop in (3, 4, 5):
            value = [x * multiplier for x in range(stop)]
            rows.append(example(
                f"What does `[x * {multiplier} for x in range({stop})]` evaluate to?",
                repr(value)
            ))
    errors = [
        ("if total = 8:", "if total == 8:", "Use `==` for equality comparison."),
        ("if name = 'Mia':", "if name == 'Mia':", "Use `==` for equality comparison."),
        ("for item items:", "for item in items:", "A for loop needs the `in` keyword."),
        ("if ready print('go')", "if ready: print('go')", "The condition needs a colon."),
    ]
    for bad, good, why in errors:
        rows.append(example(f"Correct this Python code: `{bad}`", f"{why} `{good}`"))
    return rows


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    batches = {
        "arithmetic_reasoning_scaled.jsonl": arithmetic_rows(),
        "structured_instruction_following_scaled.jsonl": structured_rows(),
        "logic_dates_spatial_calibration_scaled.jsonl": logic_rows(),
        "language_transformation_scaled.jsonl": language_rows(),
        "python_semantics_scaled.jsonl": coding_rows(),
    }
    rng = random.Random(7321)
    for name, rows in batches.items():
        rows = expand_prompt_phrasings(rows)
        rng.shuffle(rows)
        write_jsonl(name, rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted while generating capability SFT data.")
        sys.exit(130)
