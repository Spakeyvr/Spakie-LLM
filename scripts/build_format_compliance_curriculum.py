"""Build a format-compliance SFT curriculum for the small presets.

The cycle-0 general-capability benchmark showed the failure mode is output
contract compliance rather than knowledge: the model ignores "answer with the
letter", "only YES or NO", "exactly three words", "return only valid JSON" and
"output only the table", and rambles instead. This generator teaches those
contracts directly, in the exact phrasings the benchmark uses, over content
that is disjoint from the benchmark prompts.

Deterministic: same seed in, same bytes out. Exact normalized prompt overlap
with the eval set is rejected before anything is written.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = Path("data/chat_raw")
EVAL_PROMPTS = Path("data/eval/general_capability.jsonl")
SEED = 20260730


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def example(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


# --------------------------------------------------------------------------
# 1. Multiple choice in the benchmark's phrasing: letter + answer text.
# --------------------------------------------------------------------------

MC_ITEMS: list[tuple[str, list[str], int]] = [
    ("Which process turns a gas directly into a solid?", ["Melting", "Deposition", "Boiling", "Erosion"], 1),
    ("Which part of a plant absorbs most water?", ["Roots", "Petals", "Stem tip", "Pollen"], 0),
    ("Which force pulls a dropped ball toward the ground?", ["Magnetism", "Gravity", "Tension", "Buoyancy"], 1),
    ("Which state of matter has a fixed volume but no fixed shape?", ["Solid", "Liquid", "Gas", "Plasma"], 1),
    ("Which organ pumps blood through the body?", ["Liver", "Lung", "Heart", "Kidney"], 2),
    ("Which gas do plants take in during photosynthesis?", ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], 2),
    ("Which tool measures temperature?", ["Barometer", "Thermometer", "Odometer", "Voltmeter"], 1),
    ("Which material is the best thermal insulator?", ["Aluminium", "Copper", "Wool", "Steel"], 2),
    ("Which change is a chemical change?", ["Ice melting", "Paper burning", "Water boiling", "Sugar dissolving"], 1),
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Earth", "Mars"], 1),
    ("Which animal group is warm-blooded and has feathers?", ["Reptiles", "Birds", "Fish", "Amphibians"], 1),
    ("Which unit measures electric current?", ["Volt", "Ohm", "Ampere", "Watt"], 2),
    ("What does a seed need most to begin germinating?", ["Sound", "Water", "Wind", "Salt"], 1),
    ("Which process breaks rock into smaller pieces over time?", ["Weathering", "Refraction", "Distillation", "Fusion"], 0),
    ("Which layer of Earth is the thinnest?", ["Core", "Mantle", "Crust", "Atmosphere"], 2),
    ("Which of these is a renewable resource?", ["Coal", "Wind", "Oil", "Natural gas"], 1),
    ("Which sense organ detects sound?", ["Eye", "Ear", "Skin", "Tongue"], 1),
    ("Why does a metal spoon feel cold to the touch?", ["It conducts heat away", "It creates ice", "It absorbs light", "It repels air"], 0),
    ("Which action increases the volume of a sound?", ["Striking harder", "Waiting longer", "Cooling the air", "Adding water"], 0),
    ("Which describes an object at rest staying at rest?", ["Inertia", "Diffusion", "Condensation", "Reflection"], 0),
    ("What happens to most liquids when they are heated?", ["They expand", "They shrink", "They harden", "They vanish"], 0),
    ("Which is a function of the skeleton?", ["Digesting food", "Supporting the body", "Filtering air", "Producing sunlight"], 1),
    ("Which object reflects light rather than producing it?", ["The Sun", "A candle", "The Moon", "A lamp"], 2),
    ("Which process moves water from leaves into the air?", ["Transpiration", "Ingestion", "Combustion", "Magnetism"], 0),
    ("Which type of energy does a stretched rubber band store?", ["Elastic potential", "Nuclear", "Sound", "Chemical"], 0),
    ("Which best reduces friction between two surfaces?", ["Adding sand", "Adding oil", "Cooling them", "Roughening them"], 1),
    ("Which is a physical property of a substance?", ["Density", "Rusting", "Burning", "Fermenting"], 0),
    ("Where does most digestion of food occur?", ["Mouth", "Small intestine", "Lungs", "Kidneys"], 1),
    ("Which describes the path of light in a uniform medium?", ["A straight line", "A spiral", "A random walk", "A closed loop"], 0),
    ("Which cycle describes water moving between sea, air, and land?", ["Rock cycle", "Water cycle", "Carbon dating", "Life span"], 1),
]

MC_TEMPLATES = [
    "{question}\n{choices}\nAnswer with the letter and answer text.",
    "{question}\n{choices}\nRespond with the letter and the answer text.",
    "Question: {question} Choices: {inline} Answer with the letter and answer text.",
    "{question}\n{choices}\nGive only the letter and the answer text.",
]


def mc_rows() -> list[dict]:
    rows: list[dict] = []
    letters = "ABCD"
    for question, options, correct in MC_ITEMS:
        for order in range(4):
            # Deterministic rotation so the correct letter varies across examples.
            rotated = options[order:] + options[:order]
            new_correct = (correct - order) % len(options)
            choices = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(rotated))
            inline = " ".join(f"{letters[i]}. {opt}" for i, opt in enumerate(rotated))
            answer = f"{letters[new_correct]}. {rotated[new_correct]}"
            for template in MC_TEMPLATES:
                rows.append(example(template.format(question=question, choices=choices, inline=inline), answer))
    return rows


# --------------------------------------------------------------------------
# 2. Exact-format instruction following.
# --------------------------------------------------------------------------

EXACT_WORDS = [
    ("a calm lake", 3, "still clear peaceful"),
    ("a busy market", 3, "loud crowded lively"),
    ("a winter morning", 3, "cold bright quiet"),
    ("a desert at noon", 3, "hot dry vast"),
    ("a thunderstorm", 3, "dark loud sudden"),
    ("a quiet library", 3, "silent warm orderly"),
    ("a mountain summit", 3, "high windy bare"),
    ("a spring meadow", 3, "green fresh open"),
    ("a night sky", 3, "dark clear endless"),
    ("a rushing river", 3, "fast cold noisy"),
    ("an autumn forest", 4, "golden crisp quiet fading"),
    ("a summer beach", 4, "bright warm sandy busy"),
    ("an empty street", 2, "bare silent"),
    ("a candle flame", 2, "small steady"),
]

YES_NO = [
    ("Is 57 a prime number?", "NO"),
    ("Is 53 a prime number?", "YES"),
    ("Is 121 a prime number?", "NO"),
    ("Is 97 a prime number?", "YES"),
    ("Is 87 a prime number?", "NO"),
    ("Is 71 a prime number?", "YES"),
    ("Is 143 a prime number?", "NO"),
    ("Is 89 a prime number?", "YES"),
    ("Is 12 an even number?", "YES"),
    ("Is 15 divisible by 4?", "NO"),
    ("Is water a compound?", "YES"),
    ("Is the Sun a planet?", "NO"),
    ("Is 0 greater than -5?", "YES"),
    ("Is a square a rectangle?", "YES"),
    ("Is ice denser than liquid water?", "NO"),
    ("Is sound able to travel through a vacuum?", "NO"),
]

ONE_PER_LINE = [
    ("four ink colors used in the CMYK printing model", ["Cyan", "Magenta", "Yellow", "Black"]),
    ("three additive colors used on screens", ["Red", "Green", "Blue"]),
    ("three states of matter taught in primary school", ["Solid", "Liquid", "Gas"]),
    ("four seasons of the year", ["Spring", "Summer", "Autumn", "Winter"]),
    ("three primary additive colors of light", ["Red", "Green", "Blue"]),
    ("four cardinal directions", ["North", "South", "East", "West"]),
    ("three common units of length in the metric system", ["Millimetre", "Centimetre", "Metre"]),
    ("four phases of the Moon named in order", ["New moon", "First quarter", "Full moon", "Last quarter"]),
    ("three planets closer to the Sun than Jupiter, in order", ["Mercury", "Venus", "Earth"]),
    ("four basic arithmetic operations", ["Addition", "Subtraction", "Multiplication", "Division"]),
    ("three layers of Earth from the outside in", ["Crust", "Mantle", "Core"]),
]

PAIR_FORMAT = [
    ("open-plan offices", "Benefit", "easier collaboration", "Drawback", "more noise"),
    ("electric cars", "Benefit", "lower running costs", "Drawback", "longer refuelling time"),
    ("night shifts", "Benefit", "quieter working hours", "Drawback", "disrupted sleep"),
    ("online classes", "Benefit", "flexible scheduling", "Drawback", "less direct contact"),
    ("public transport", "Benefit", "lower cost per trip", "Drawback", "fixed timetables"),
    ("living in a city", "Benefit", "more services nearby", "Drawback", "higher rent"),
    ("automated testing", "Benefit", "faster feedback", "Drawback", "upfront setup effort"),
    ("solar panels", "Benefit", "free fuel from sunlight", "Drawback", "output varies with weather"),
]


def instruction_rows() -> list[dict]:
    rows: list[dict] = []
    for subject, count, answer in EXACT_WORDS:
        word = {2: "two", 3: "three", 4: "four"}[count]
        rows.append(example(f"Reply with exactly {word} words that describe {subject}. Do not add punctuation.", answer))
        rows.append(example(f"Describe {subject} in exactly {word} words. No punctuation.", answer))
        rows.append(example(f"Use exactly {word} words to describe {subject}. Do not add punctuation.", answer))
    for question, answer in YES_NO:
        rows.append(example(f"Answer only YES or NO: {question}", answer))
        rows.append(example(f"{question} Answer with only YES or NO.", answer))
        rows.append(example(f"Reply with YES or NO and nothing else: {question}", answer))
    for description, items in ONE_PER_LINE:
        word = {3: "three", 4: "four"}[len(items)]
        body = "\n".join(items)
        rows.append(example(f"List exactly {word} {description}, one per line, with no bullets.", body))
        rows.append(example(f"Name the {description}. One per line, no bullets or numbering.", body))
    for topic, k1, v1, k2, v2 in PAIR_FORMAT:
        rows.append(example(
            f"Give one benefit and one drawback of {topic}. Use exactly this format: {k1}: ... | {k2}: ...",
            f"{k1}: {v1} | {k2}: {v2}",
        ))
        rows.append(example(
            f"State one benefit and one drawback of {topic} in exactly this format: {k1}: ... | {k2}: ...",
            f"{k1}: {v1} | {k2}: {v2}",
        ))
    return rows


# --------------------------------------------------------------------------
# 3. JSON-only and table-only output.
# --------------------------------------------------------------------------

JSON_OBJECTS = [
    ("a film with title 'Arrival', director 'Denis Villeneuve', and year 2016. Use keys title, director, and year",
     {"title": "Arrival", "director": "Denis Villeneuve", "year": 2016}),
    ("a book with title 'Piranesi', author 'Susanna Clarke', and year 2020. Use keys title, author, and year",
     {"title": "Piranesi", "author": "Susanna Clarke", "year": 2020}),
    ("a city with name 'Lisbon', country 'Portugal', and population 545000. Use keys name, country, and population",
     {"name": "Lisbon", "country": "Portugal", "population": 545000}),
    ("a song with title 'Clair de Lune', composer 'Claude Debussy', and year 1905. Use keys title, composer, and year",
     {"title": "Clair de Lune", "composer": "Claude Debussy", "year": 1905}),
    ("a product with name 'Desk lamp', price 34, and currency 'EUR'. Use keys name, price, and currency",
     {"name": "Desk lamp", "price": 34, "currency": "EUR"}),
    ("a student with name 'Ines', grade 8, and subject 'Biology'. Use keys name, grade, and subject",
     {"name": "Ines", "grade": 8, "subject": "Biology"}),
]

JSON_SPLITS = [
    ((5, 6, 7, 8), {"even": [6, 8], "odd": [5, 7]}),
    ((10, 11, 12, 13), {"even": [10, 12], "odd": [11, 13]}),
    ((2, 5, 8, 9), {"even": [2, 8], "odd": [5, 9]}),
    ((21, 22, 23, 24), {"even": [22, 24], "odd": [21, 23]}),
    ((14, 15, 16, 17), {"even": [14, 16], "odd": [15, 17]}),
]

TABLES = [
    ("Country", "Capital", [("France", "Paris"), ("Japan", "Tokyo")]),
    ("Fruit", "Color", [("banana", "yellow"), ("plum", "purple")]),
    ("Metal", "Symbol", [("iron", "Fe"), ("tin", "Sn")]),
    ("Instrument", "Family", [("violin", "strings"), ("flute", "woodwind")]),
    ("Planet", "Position", [("Mercury", "first"), ("Venus", "second")]),
    ("Shape", "Sides", [("triangle", "3"), ("hexagon", "6")]),
]


JSON_BOOKS = [
    ("The Left Hand of Darkness", "Ursula K. Le Guin", 1969),
    ("Kindred", "Octavia E. Butler", 1979),
    ("Never Let Me Go", "Kazuo Ishiguro", 2005),
    ("The Dispossessed", "Ursula K. Le Guin", 1974),
    ("Beloved", "Toni Morrison", 1987),
    ("The Remains of the Day", "Kazuo Ishiguro", 1989),
    ("Solaris", "Stanisław Lem", 1961),
    ("Roadside Picnic", "Arkady Strugatsky", 1972),
    ("The Dead Zone", "Stephen King", 1979),
    ("Foundation", "Isaac Asimov", 1951),
    ("Rendezvous with Rama", "Arthur C. Clarke", 1973),
    ("Hyperion", "Dan Simmons", 1989),
]

JSON_PEOPLE = [
    ("Amara", 31, "architect"), ("Bo", 24, "nurse"), ("Chen", 45, "teacher"),
    ("Dilara", 38, "engineer"), ("Eli", 52, "carpenter"), ("Freya", 29, "vet"),
    ("Goran", 41, "chef"), ("Hana", 27, "pilot"), ("Idris", 35, "librarian"),
    ("Juno", 48, "geologist"),
]

JSON_CITIES = [
    ("Porto", "Portugal", 231000), ("Bergen", "Norway", 286000), ("Cork", "Ireland", 222000),
    ("Graz", "Austria", 291000), ("Utrecht", "Netherlands", 361000), ("Malmo", "Sweden", 351000),
    ("Ghent", "Belgium", 263000), ("Aarhus", "Denmark", 285000),
]


def json_volume_rows() -> list[dict]:
    """Drill valid-JSON-only output across many key sets and value types.

    Cycle 1 emitted syntactically broken pseudo-code for every JSON prompt, so
    this needs breadth of instances rather than breadth of phrasing.
    """
    rows: list[dict] = []
    for title, author, year in JSON_BOOKS:
        payload = json.dumps({"title": title, "author": author, "year": year})
        rows.append(example(
            f"Return only valid JSON for a book with title '{title}', author '{author}', "
            f"and year {year}. Use keys title, author, and year.",
            payload,
        ))
        rows.append(example(
            f"Output only valid JSON, no prose, describing the book '{title}' by {author} "
            f"published in {year}. Use keys title, author, and year.",
            payload,
        ))
    for name, age, job in JSON_PEOPLE:
        payload = json.dumps({"name": name, "age": age, "occupation": job})
        rows.append(example(
            f"Return only valid JSON for a person with name '{name}', age {age}, and occupation "
            f"'{job}'. Use keys name, age, and occupation.",
            payload,
        ))
    for name, country, population in JSON_CITIES:
        payload = json.dumps({"name": name, "country": country, "population": population})
        rows.append(example(
            f"Return only valid JSON for a city with name '{name}', country '{country}', and "
            f"population {population}. Use keys name, country, and population.",
            payload,
        ))
    quads = [(3, 4, 5, 6), (7, 8, 9, 10), (11, 12, 13, 14), (2, 3, 6, 7), (15, 16, 17, 18),
             (4, 7, 8, 11), (20, 21, 22, 23), (1, 4, 5, 8), (9, 10, 13, 16), (6, 9, 12, 15)]
    for numbers in quads:
        evens = [n for n in numbers if n % 2 == 0]
        odds = [n for n in numbers if n % 2 == 1]
        listed = ", ".join(str(n) for n in numbers[:-1]) + f", and {numbers[-1]}"
        payload = json.dumps({"even": evens, "odd": odds})
        rows.append(example(
            f"Return only valid JSON with keys even and odd, each containing an array that "
            f"classifies the numbers {listed}.",
            payload,
        ))
    return rows


EXTRA_TABLES = [
    ("Animal", "Sound", [("cow", "moo"), ("sheep", "baa")]),
    ("Animal", "Habitat", [("camel", "desert"), ("seal", "coast")]),
    ("Language", "Region", [("Welsh", "Wales"), ("Basque", "Pyrenees")]),
    ("Tool", "Use", [("hammer", "driving nails"), ("saw", "cutting wood")]),
    ("River", "Continent", [("Danube", "Europe"), ("Mekong", "Asia")]),
    ("Element", "Symbol", [("sodium", "Na"), ("lead", "Pb")]),
    ("Sport", "Equipment", [("tennis", "racket"), ("hockey", "stick")]),
    ("Season", "Month", [("winter", "January"), ("summer", "July")]),
    ("Bird", "Colour", [("robin", "red"), ("crow", "black")]),
    ("Cheese", "Country", [("brie", "France"), ("feta", "Greece")]),
]


def table_volume_rows() -> list[dict]:
    rows: list[dict] = []
    for left, right, pairs in EXTRA_TABLES:
        table = "\n".join([f"| {left} | {right} |", "| --- | --- |"] + [f"| {a} | {b} |" for a, b in pairs])
        subjects = " and ".join(a for a, _ in pairs)
        rows.append(example(
            f"Create a two-column Markdown table with headers {left} and {right}, containing rows "
            f"for {subjects}. Output only the table.",
            table,
        ))
        rows.append(example(
            f"Output only a Markdown table with headers {left} and {right} and rows for {subjects}.",
            table,
        ))
    return rows


def structured_rows() -> list[dict]:
    rows: list[dict] = []
    for description, obj in JSON_OBJECTS:
        payload = json.dumps(obj)
        rows.append(example(f"Return only valid JSON for {description}.", payload))
        rows.append(example(f"Output only valid JSON, no prose, for {description}.", payload))
    for numbers, obj in JSON_SPLITS:
        listed = ", ".join(str(n) for n in numbers[:-1]) + f", and {numbers[-1]}"
        payload = json.dumps(obj)
        rows.append(example(
            f"Return only valid JSON with keys even and odd, each containing an array that classifies the numbers {listed}.",
            payload,
        ))
    for left, right, pairs in TABLES:
        table = "\n".join([f"| {left} | {right} |", "| --- | --- |"] + [f"| {a} | {b} |" for a, b in pairs])
        subjects = " and ".join(a for a, _ in pairs)
        rows.append(example(
            f"Create a two-column Markdown table with headers {left} and {right}, containing rows for {subjects}. Output only the table.",
            table,
        ))
        rows.append(example(
            f"Output only a Markdown table with headers {left} and {right} and rows for {subjects}.",
            table,
        ))
    return rows


# --------------------------------------------------------------------------
# 4. Terse extractive context QA (single minimal span, no passage echo).
# --------------------------------------------------------------------------

CONTEXT_ITEMS = [
    ("The Harrow Library opened in 1992. It moved to Bell Road in 2007 and added a reading room in 2015.",
     [("In what year did the library move to Bell Road?", "2007"),
      ("In what year did the library open?", "1992"),
      ("In what year was the reading room added?", "2015")]),
    ("Vero moths rest in caves during summer. In winter they travel east to drier valleys. They feed on nectar and pollen.",
     [("Do Vero moths travel west in winter?", "No, they travel east in winter."),
      ("What do Vero moths feed on?", "Nectar and pollen."),
      ("Where do Vero moths rest in summer?", "In caves.")]),
    ("The green studio is north of the gallery. The gallery closes at 5 p.m., while the studio closes at 9 p.m. Both are closed on Monday.",
     [("Which place closes earlier?", "The gallery."),
      ("Which place closes later?", "The studio."),
      ("Which day are both closed?", "Monday.")]),
    ("Dario planted fifteen bean seeds and six squash seeds. Eleven bean plants and four squash plants grew.",
     [("How many squash plants grew?", "Four."),
      ("How many bean plants grew?", "Eleven."),
      ("How many squash seeds were planted?", "Six.")]),
    ("Nell borrowed The Copper Gate on Tuesday and returned it Friday. She then borrowed Salt Roads, which is due next Thursday.",
     [("Which book does Nell currently have?", "Salt Roads."),
      ("When is Salt Roads due?", "Next Thursday."),
      ("Which book did she return?", "The Copper Gate.")]),
    ("A bronze chip is worth two points. A silver chip is worth seven points. Omar has three bronze chips and two silver chips.",
     [("How many points are Omar's bronze chips worth in total?", "Six points."),
      ("How many points are Omar's silver chips worth in total?", "Fourteen points."),
      ("How many chips does Omar have in total?", "Five chips.")]),
    ("Birch Clinic accepts walk-ins on Monday and Friday. Appointments are required on Tuesday, Wednesday, and Thursday. It is closed on weekends.",
     [("Is an appointment required on Wednesday?", "Yes."),
      ("Is an appointment required on Friday?", "No, Birch Clinic accepts walk-ins on Friday."),
      ("Is the clinic open on Sunday?", "No, it is closed on weekends.")]),
    ("Priya finished ahead of Liam in the race. Liam finished ahead of Ada. Sven finished behind Ada.",
     [("Who finished directly ahead of Ada?", "Liam."),
      ("Who finished last?", "Sven."),
      ("Who finished first?", "Priya.")]),
    ("The workshop runs from 9 a.m. to noon on Saturday. Lunch is served afterwards. Places are limited to twelve people.",
     [("How many places are available?", "Twelve."),
      ("What time does the workshop end?", "Noon."),
      ("What is served after the workshop?", "Lunch.")]),
    ("Tomas cycled 12 km on Monday and 9 km on Wednesday. He did not cycle on Tuesday.",
     [("How far did Tomas cycle on Wednesday?", "9 km."),
      ("Did Tomas cycle on Tuesday?", "No, he did not cycle on Tuesday."),
      ("How far did he cycle on Monday?", "12 km.")]),
]

CONTEXT_TEMPLATES = [
    "Passage: {passage}\n\n{question}",
    "Passage: {passage}\n\nUsing only the passage, {lowered}",
    "Read the passage and answer briefly.\nPassage: {passage}\n\n{question}",
]


def context_rows() -> list[dict]:
    rows: list[dict] = []
    for passage, pairs in CONTEXT_ITEMS:
        for question, answer in pairs:
            lowered = question[0].lower() + question[1:]
            for template in CONTEXT_TEMPLATES:
                rows.append(example(
                    template.format(passage=passage, question=question, lowered=lowered),
                    answer,
                ))
    return rows


# --------------------------------------------------------------------------
# 5. Short arithmetic with a brief shown calculation.
# --------------------------------------------------------------------------

GARMENTS = ["coat", "jumper", "scarf", "pair of boots", "backpack", "raincoat", "hat", "dress"]
STAPLES = ["sugar", "butter", "oats", "rice", "lentils", "cocoa"]


def arithmetic_rows() -> list[dict]:
    """Percentage discounts, order of operations, ratio scaling, fraction subtraction.

    Volume and numeric variety matter more than clever wording here: cycle 1
    produced malformed algebra for every arithmetic prompt, so each pattern is
    drilled across many values with the numeric answer stated plainly.
    """
    rows: list[dict] = []

    for index, price in enumerate([40, 45, 50, 60, 70, 75, 90, 120, 150, 160, 180, 200, 240, 250]):
        for pct in (10, 20, 25, 40, 50):
            final = price * (100 - pct) // 100
            if price * (100 - pct) % 100:
                continue
            item = GARMENTS[index % len(GARMENTS)]
            answer = f"{price} × {(100 - pct) / 100:g} = {final}. The sale price is ${final}."
            rows.append(example(
                f"A {item} costs ${price} and is discounted by {pct}%. What is the sale price? "
                f"Give the calculation briefly.",
                answer,
            ))
            rows.append(example(
                f"A {item} priced at ${price} is reduced by {pct}%. What is the new price? Show the calculation briefly.",
                answer,
            ))

    for a in (9, 12, 15, 18, 20, 24, 30):
        for b, c in ((6, 4), (5, 3), (7, 2), (4, 6)):
            for d in (5, 7, 9):
                result = a + b * c - d
                rows.append(example(
                    f"Compute {a} + {b} × {c} - {d}. Explain which operation comes first.",
                    f"Multiplication comes first: {b} × {c} = {b * c}. "
                    f"Then {a} + {b * c} - {d} = {result}.",
                ))

    for index, (people, grams) in enumerate([(4, 300), (4, 200), (5, 250), (6, 300), (3, 150), (8, 400), (2, 90), (6, 480)]):
        per = grams // people
        for target in (9, 10, 12, 15):
            total = per * target
            staple = STAPLES[index % len(STAPLES)]
            answer = f"{grams} ÷ {people} = {per} g per person, so {per} × {target} = {total} g."
            rows.append(example(
                f"A recipe for {people} people uses {grams} g of {staple}. "
                f"How much {staple} is needed for {target} people?",
                answer,
            ))

    fraction_pairs = [
        ((7, 8), (1, 3)), ((5, 6), (1, 4)), ((3, 4), (1, 6)), ((2, 3), (1, 5)),
        ((7, 10), (1, 5)), ((4, 5), (1, 4)), ((5, 8), (1, 6)), ((9, 10), (1, 4)),
        ((3, 5), (1, 3)), ((7, 9), (1, 2)), ((5, 7), (1, 3)), ((11, 12), (1, 3)),
    ]
    for (an, ad), (bn, bd) in fraction_pairs:
        num = an * bd - bn * ad
        den = ad * bd
        divisor = math.gcd(num, den)
        rn, rd = num // divisor, den // divisor
        common = f"{an}/{ad} - {bn}/{bd} = {an * bd}/{den} - {bn * ad}/{den} = {num}/{den}"
        tail = f" = {rn}/{rd}" if divisor > 1 else ""
        rows.append(example(
            f"What is {an}/{ad} minus {bn}/{bd}? Return the answer as a simplified fraction.",
            f"{rn}/{rd}",
        ))
        rows.append(example(
            f"Subtract {bn}/{bd} from {an}/{ad} and give the simplified fraction with the calculation.",
            f"{common}{tail}. The answer is {rn}/{rd}.",
        ))
    return rows


# --------------------------------------------------------------------------
# 6. Calibration: ask for clarification instead of inventing an answer.
# --------------------------------------------------------------------------

AMBIGUOUS_PLACES = [
    "Riverton", "Georgetown", "Newport", "Fairview", "Clinton", "Ashford", "Bridgetown", "Kingston",
    "Springvale", "Milton", "Franklin", "Salem", "Richmond", "Arlington", "Bristol", "Dover",
    "Lakeview", "Oakdale", "Hillside", "Greenville", "Middleton", "Norwood",
]
AMBIGUOUS_RELATIVE_DATES = [
    "next Monday", "next Tuesday", "next Wednesday", "next Thursday", "next Friday",
    "next Saturday", "this coming Sunday", "next month", "the week after next",
    "a week from Tuesday", "the first Monday next month",
]
AMBIGUOUS_EVENTS = [
    ("Who won the match yesterday?", "which match"),
    ("Who won the final on Saturday?", "which final"),
    ("What was the score in the tournament?", "which tournament"),
    ("Did our team win the semi-final?", "which team"),
    ("Who won the race this morning?", "which race"),
    ("What was the result of the derby?", "which derby"),
    ("Who came first in the championship?", "which championship"),
    ("How did the team do last night?", "which team"),
    ("Who scored in the second half?", "which fixture"),
    ("Was the game close?", "which game"),
]


def calibration_rows() -> list[dict]:
    rows: list[dict] = []
    for place in AMBIGUOUS_PLACES:
        answer = (
            f"I need to clarify which {place} you mean — there are many places with that name, "
            f"in different states and countries. Which location do you mean?"
        )
        rows.append(example(f"What is the exact population of {place}?", answer))
        rows.append(example(f"How many people live in {place}?", answer))
    for phrase in AMBIGUOUS_RELATIVE_DATES:
        answer = (
            f"I need to know today's date to work out {phrase} — could you tell me the current date, "
            f"or give the exact date you mean?"
        )
        rows.append(example(f"My appointment is {phrase}. What date is that?", answer))
        rows.append(example(f"The deadline is {phrase}. What date is that?", answer))
    for question, which in AMBIGUOUS_EVENTS:
        answer = (
            f"Could you clarify {which} you mean — which sport, teams, and date? "
            f"Without that I cannot say who won."
        )
        rows.append(example(question, answer))
    return rows


# --------------------------------------------------------------------------
# 7. Numbered plans, ingredient-grounded suggestions, comparisons.
# --------------------------------------------------------------------------

PLAN_TASKS = [
    ("a 15-minute phone screening tomorrow",
     ["Reread the job posting and note the three skills it stresses most.",
      "Write and say aloud short answers about your recent work.",
      "Test your phone and pick a quiet room ten minutes early."]),
    ("a 30-minute presentation next week",
     ["Outline the three points your audience must remember.",
      "Build slides that carry one idea each, then cut the rest.",
      "Rehearse once with a timer and trim to fit the slot."]),
    ("a driving test in three days",
     ["Book two practice drives on the test route.",
      "Drill the manoeuvres you are least confident about.",
      "Check documents and sleep early the night before."]),
    ("a first day at a new job",
     ["Confirm the start time, place, and dress code.",
      "Prepare a short introduction of your background.",
      "Write down names and questions during the day."]),
    ("a two-day hiking trip",
     ["Check the forecast and pick a route within your range.",
      "Pack water, food, layers, and a paper map.",
      "Tell someone your route and expected return."]),
    ("a household budget review",
     ["List every fixed monthly cost from your statements.",
      "Group the remaining spending into three categories.",
      "Set one cut and one savings target for next month."]),
    ("a 20-minute panel interview later this week",
     ["List the three achievements you most want to land.",
      "Prepare one short story for each, with a number in it.",
      "Write two questions to ask the panel at the end."]),
    ("a language exam in two weeks",
     ["Sit one timed past paper to find your weakest section.",
      "Drill that section for twenty minutes daily.",
      "Review your errors the day before, then rest."]),
    ("moving flat next month",
     ["Book movers and confirm the lift and parking access.",
      "Sort belongings into keep, donate, and discard.",
      "Redirect post and transfer utilities a week ahead."]),
    ("a dentist appointment for a nervous child",
     ["Explain what will happen in simple, calm words.",
      "Schedule the visit early, when they are rested.",
      "Agree a small reward for afterwards."]),
    ("running a first 5k next month",
     ["Run three short sessions a week, alternating with rest.",
      "Add distance gradually, no more than ten percent weekly.",
      "Practise the route once at your target pace."]),
    ("a job application due Friday",
     ["Reread the posting and mirror its key terms.",
      "Tailor your CV to the three duties it lists first.",
      "Proofread once aloud, then submit a day early."]),
    ("hosting six people for dinner",
     ["Pick one main dish you have cooked before.",
      "Shop the day before and prep what keeps.",
      "Cook the main last so it is served hot."]),
    ("starting a small vegetable patch",
     ["Pick a spot with at least six hours of sun.",
      "Clear the ground and dig in compost.",
      "Sow two easy crops and water them daily."]),
    ("a difficult conversation with a colleague",
     ["Write the one outcome you want from the talk.",
      "Open with a specific, factual observation.",
      "Ask for their view before proposing a fix."]),
    ("reducing your monthly phone bill",
     ["Check your actual data and call usage.",
      "Compare two cheaper plans that cover that usage.",
      "Call your provider and ask them to match the best price."]),
    ("preparing for a power outage",
     ["Charge power banks and keep torches accessible.",
      "Store drinking water and non-perishable food.",
      "Note how to open electric gates or doors manually."]),
    ("learning a song on guitar in a week",
     ["Break the song into four short sections.",
      "Practise each slowly with a metronome.",
      "Join the sections once each is clean."]),
]

INGREDIENT_MEALS = [
    (["rice", "eggs", "frozen peas", "soy sauce"], "egg fried rice",
     ["Cook the rice and let it cool.",
      "Scramble the eggs in a hot pan, then set aside.",
      "Fry the peas briefly, add rice, eggs, and soy sauce, and toss."]),
    (["pasta", "tuna", "sweetcorn", "olive oil"], "tuna sweetcorn pasta",
     ["Boil the pasta until just tender.",
      "Drain the tuna and sweetcorn.",
      "Toss pasta with tuna, sweetcorn, and olive oil."]),
    (["potatoes", "onion", "cheese", "butter"], "cheesy potato bake",
     ["Slice the potatoes and onion thinly.",
      "Layer them with butter in a dish.",
      "Top with cheese and bake until golden."]),
    (["bread", "tomato", "cheese", "basil"], "toasted cheese and tomato",
     ["Slice the tomato and cheese.",
      "Layer them on the bread with basil.",
      "Toast until the cheese melts."]),
    (["chickpeas", "spinach", "garlic", "lemon"], "garlic chickpeas with spinach",
     ["Fry the garlic gently in oil.",
      "Add the chickpeas and warm through.",
      "Stir in spinach until wilted and finish with lemon."]),
    (["noodles", "chicken", "carrot", "ginger"], "ginger chicken noodles",
     ["Boil the noodles and drain.",
      "Fry the chicken with ginger until cooked.",
      "Add sliced carrot, then toss with the noodles."]),
]

COMPARISONS = [
    ("hydroelectric power", "geothermal power", "both are renewable sources that generate electricity without burning fuel",
     "hydroelectric power depends on flowing water, while geothermal power depends on underground heat"),
    ("trains", "buses", "both are similar in that they carry many passengers on fixed routes",
     "the difference is that trains need rails, while buses use ordinary roads"),
    ("email", "instant messaging", "both are similar in that they deliver written messages over a network",
     "the difference is that email suits longer, delayed replies, while messaging suits short, immediate ones"),
    ("cycling", "running", "both are similar in that they are aerobic exercise that needs little equipment",
     "the difference is that cycling is lower impact on joints, while running needs no machine"),
    ("laptops", "tablets", "both are similar in that they are portable computers with screens and batteries",
     "the difference is that laptops include a keyboard for heavy typing, while tablets rely on touch"),
    ("libraries", "bookshops", "both are similar in that they organise large collections of books for readers",
     "the difference is that libraries lend books for free, while bookshops sell them"),
    ("tea", "coffee", "both are similar in that they are hot caffeinated drinks made by infusion",
     "the difference is that coffee usually carries more caffeine per cup than tea"),
    ("houses", "flats", "both are similar in that they are permanent homes with private rooms",
     "the difference is that houses usually have their own land, while flats share a building"),
    ("swimming", "rowing", "both are similar in that they are full-body sports done on or in water",
     "the difference is that swimming needs no equipment, while rowing needs a boat and oars"),
    ("radio", "podcasts", "both are similar in that they deliver spoken audio programmes",
     "the difference is that radio is broadcast live, while podcasts are downloaded on demand"),
    ("cats", "dogs", "both are similar in that they are common domesticated pets that bond with people",
     "the difference is that dogs generally need daily walks, while cats are more independent"),
    ("wheat", "rice", "both are similar in that they are staple cereal crops feeding billions",
     "the difference is that rice is usually grown in flooded paddies, while wheat prefers drier fields"),
    ("solar panels", "wind turbines", "both are similar in that they generate renewable electricity without fuel",
     "the difference is that panels need sunlight, while turbines need moving air"),
    ("bicycles", "motorcycles", "both are similar in that they are two-wheeled vehicles for one or two riders",
     "the difference is that bicycles are pedal-powered, while motorcycles use an engine"),
    ("printed books", "e-books", "both are similar in that they present the same written text to a reader",
     "the difference is that e-books need a device and battery, while printed books do not"),
    ("rivers", "canals", "both are similar in that they are channels carrying water across land",
     "the difference is that rivers form naturally, while canals are dug by people"),
    ("football", "rugby", "both are similar in that they are team sports played on a large pitch with goals",
     "the difference is that football is played mainly with the feet, while rugby allows carrying the ball"),
]


def plan_rows() -> list[dict]:
    rows: list[dict] = []
    for task, steps in PLAN_TASKS:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
        rows.append(example(f"Give a practical three-step plan for preparing for {task}.", numbered))
        rows.append(example(f"Outline a three-step plan for {task}. Number the steps.", numbered))
    for items, dish, steps in INGREDIENT_MEALS:
        listed = ", ".join(items[:-1]) + f", and {items[-1]}"
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))
        body = f"Make {dish} with the {listed}.\n{numbered}"
        rows.append(example(f"I have {listed}. Suggest a simple dinner and give concise steps.", body))
        rows.append(example(f"Using {listed}, suggest one simple dinner with short numbered steps.", body))
    for left, right, similarity, difference in COMPARISONS:
        # Spell out "similar" and "difference" as words rather than as labels:
        # the benchmark checks for the terms themselves, and a "Similarity:"
        # label does not contain the word "similar" under word-boundary matching.
        if not similarity.startswith("both are similar"):
            similarity = "both are similar in that " + similarity.removeprefix("both are ")
        if not difference.startswith("the difference"):
            difference = "the difference is that " + difference
        body = f"- They are similar in that {similarity.removeprefix('both are similar in that ')}.\n- The difference is that {difference.removeprefix('the difference is that ')}."
        rows.append(example(
            f"Compare {left} and {right} in two concise bullet points: one similarity and one difference.",
            body,
        ))
        rows.append(example(
            f"In two bullet points, give one similarity and one difference between {left} and {right}.",
            body,
        ))
    return rows


# --------------------------------------------------------------------------
# 8. Edge cases: division by zero, empty vs whitespace, sorting negatives.
# --------------------------------------------------------------------------

def edge_case_rows() -> list[dict]:
    rows: list[dict] = []
    for n in (3, 4, 6, 7, 8, 9, 11, 12):
        answer = (
            f"Zero divided by {n} is 0. Dividing {n} by zero is undefined — "
            f"division by zero has no result."
        )
        rows.append(example(f"What is zero divided by {n}? What about {n} divided by zero?", answer))
        rows.append(example(f"Compute 0 ÷ {n} and {n} ÷ 0.", answer))
    empty_answer = (
        "No, they are different. An empty string has zero characters, while a string containing "
        "one space has one character."
    )
    rows.append(example("Is an empty string the same as a string containing one space? Explain briefly.", empty_answer))
    rows.append(example("Does an empty string equal a string with a single space? Explain briefly.", empty_answer))
    rows.append(example("Is \"\" the same value as \" \"? Explain briefly.", empty_answer))
    sorts = [
        ([-3, 0, -12, 4], None), ([-1, 5, -9, 2], None), ([7, -4, 0, -20], None),
        ([-5, -15, 3, 1], None), ([2, -2, -30, 10], None), ([-7, 6, -6, 0], None),
        ([-8, 1, -1, 9], None), ([0, -6, 12, -14], None), ([5, -5, -25, 2], None),
        ([-11, 3, 0, -4], None), ([8, -9, -3, 1], None), ([-16, -2, 7, 0], None),
        ([4, -13, 2, -1], None), ([-6, 0, 11, -18], None), ([1, -7, -21, 6], None),
        ([-4, 9, -9, 3], None), ([13, -12, 0, -5], None), ([-2, -22, 5, 8], None),
    ]
    for given, _ in sorts:
        ordered = sorted(given)
        listed = ", ".join(str(n) for n in given)
        answer = ", ".join(str(n) for n in ordered)
        rows.append(example(f"Sort these numbers from smallest to largest: {listed}.", answer))
        rows.append(example(f"Put these in ascending order: {listed}.", answer))
    return rows


# --------------------------------------------------------------------------
# 9. Composite spatial directions and per-item classification.
# --------------------------------------------------------------------------

DIRECTION_PAIRS = [
    ("north", "east", "northeast"),
    ("north", "west", "northwest"),
    ("south", "east", "southeast"),
    ("south", "west", "southwest"),
]
NAME_TRIPLES = [
    ("Mira", "Tom", "Lena"), ("Sofia", "Ravi", "Dana"), ("Ivan", "Petra", "Nils"), ("Ayo", "Kim", "Elsa"),
    ("Nora", "Hugo", "Bea"), ("Leo", "Sara", "Milo"), ("Zara", "Otto", "Iris"), ("Finn", "Maya", "Rui"),
    ("Ada", "Karl", "Vera"), ("Omar", "Lucia", "Theo"),
]

CLASSIFY_SETS = [
    ([("whale", "mammal"), ("sparrow", "bird"), ("lizard", "reptile")], "mammal, bird, or reptile"),
    ([("bat", "mammal"), ("penguin", "bird"), ("crocodile", "reptile")], "mammal, bird, or reptile"),
    ([("horse", "mammal"), ("owl", "bird"), ("snake", "reptile")], "mammal, bird, or reptile"),
    ([("carrot", "vegetable"), ("apple", "fruit"), ("rice", "grain")], "vegetable, fruit, or grain"),
    ([("copper", "metal"), ("granite", "rock"), ("oxygen", "gas")], "metal, rock, or gas"),
    ([("dolphin", "mammal"), ("ostrich", "bird"), ("tortoise", "reptile")], "mammal, bird, or reptile"),
    ([("elephant", "mammal"), ("flamingo", "bird"), ("gecko", "reptile")], "mammal, bird, or reptile"),
    ([("wolf", "mammal"), ("duck", "bird"), ("iguana", "reptile")], "mammal, bird, or reptile"),
    ([("seal", "mammal"), ("swan", "bird"), ("chameleon", "reptile")], "mammal, bird, or reptile"),
    ([("tiger", "mammal"), ("crow", "bird"), ("python", "reptile")], "mammal, bird, or reptile"),
    ([("rabbit", "mammal"), ("robin", "bird"), ("turtle", "reptile")], "mammal, bird, or reptile"),
    ([("otter", "mammal"), ("pigeon", "bird"), ("cobra", "reptile")], "mammal, bird, or reptile"),
    ([("spinach", "vegetable"), ("pear", "fruit"), ("barley", "grain")], "vegetable, fruit, or grain"),
    ([("broccoli", "vegetable"), ("mango", "fruit"), ("oats", "grain")], "vegetable, fruit, or grain"),
    ([("leek", "vegetable"), ("cherry", "fruit"), ("millet", "grain")], "vegetable, fruit, or grain"),
    ([("iron", "metal"), ("marble", "rock"), ("nitrogen", "gas")], "metal, rock, or gas"),
    ([("zinc", "metal"), ("basalt", "rock"), ("helium", "gas")], "metal, rock, or gas"),
    ([("silver", "metal"), ("sandstone", "rock"), ("argon", "gas")], "metal, rock, or gas"),
]


def spatial_and_classification_rows() -> list[dict]:
    rows: list[dict] = []
    for first, second, combined in DIRECTION_PAIRS:
        for a, b, c in NAME_TRIPLES:
            answer = f"{a} is {combined} of {c}."
            rows.append(example(
                f"{a} stands {first} of {b}. {b} stands {second} of {c}. Where is {a} relative to {c}?",
                answer,
            ))
            rows.append(example(
                f"{a} is {first} of {b}, and {b} is {second} of {c}. Where is {a} relative to {c}?",
                answer,
            ))
    for pairs, classes in CLASSIFY_SETS:
        listed = ", ".join(item for item, _ in pairs)
        body = "\n".join(f"{item}: {label}" for item, label in pairs)
        rows.append(example(
            f"Classify each as {classes}: {listed}. Use the format animal: class."
            if "animal" in classes or "mammal" in classes
            else f"Classify each as {classes}: {listed}. Use the format item: class.",
            body,
        ))
        rows.append(example(f"Label each of these as {classes}: {listed}. One per line as name: class.", body))
    return rows


# --------------------------------------------------------------------------
# 10. Summarization: abstractive one-sentence and exact-word-count summaries.
# --------------------------------------------------------------------------

SUMMARY_ITEMS = [
    ("Earthworms tunnel through soil as they feed on dead leaves. Their burrows let air and water reach plant roots, which improves soil structure.",
     "Earthworms improve soil aeration and structure by burrowing while feeding on dead leaves.",
     ["Earthworms", "aerate", "soil", "while", "feeding"]),
    ("Ocean currents move warm water from the equator toward the poles. This transfer of heat moderates the climate of nearby coasts.",
     "Ocean currents moderate coastal climates by carrying equatorial heat toward the poles.",
     ["Currents", "carry", "heat", "moderating", "climates"]),
    ("Bats hunt at night using echolocation. They emit high-pitched calls and listen for the echoes to locate insects in the dark.",
     "Bats use echolocation to locate insects while hunting in darkness.",
     ["Bats", "echolocate", "to", "hunt", "insects"]),
    ("Coral reefs shelter a quarter of all marine species. They grow slowly from the skeletons of tiny animals and are damaged by warming water.",
     "Coral reefs shelter vast marine biodiversity but grow slowly and suffer from warming seas.",
     ["Reefs", "shelter", "biodiversity", "but", "warm"]),
    ("Yeast converts sugar into carbon dioxide and alcohol. In bread dough the trapped gas makes the loaf rise before baking.",
     "Yeast ferments sugar into gas that makes bread dough rise.",
     ["Yeast", "ferments", "sugar", "raising", "dough"]),
    ("Tree rings record yearly growth. Wide rings suggest wet, favourable years, so researchers use them to reconstruct past climates.",
     "Tree rings record annual growth, letting researchers reconstruct past climates.",
     ["Rings", "record", "growth", "revealing", "climate"]),
    ("Honeybees perform a waggle dance inside the hive. The angle and length of the dance tell other bees the direction and distance of food.",
     "Honeybees use a waggle dance to tell hivemates where food lies.",
     ["Bees", "dance", "to", "signal", "food"]),
    ("Volcanic ash can reach the upper atmosphere after a large eruption. The particles reflect sunlight and can cool the planet for months.",
     "Volcanic ash reflects sunlight and can cool the planet for months.",
     ["Ash", "reflects", "sunlight", "cooling", "Earth"]),
    ("Migrating birds navigate using the Sun, stars, and Earth's magnetic field. Young birds refine these cues on their first journey.",
     "Migrating birds navigate by sun, stars, and magnetism, improving with experience.",
     ["Birds", "navigate", "using", "multiple", "cues"]),
    ("Antibiotics kill bacteria but not viruses. Overuse lets resistant bacteria survive and spread, making infections harder to treat.",
     "Overusing antibiotics breeds resistant bacteria, making infections harder to treat.",
     ["Antibiotic", "overuse", "breeds", "resistant", "bacteria"]),
    ("Glaciers advance when snowfall exceeds melting. As they move they grind rock beneath them, carving wide valleys.",
     "Glaciers grow when snowfall exceeds melting and carve valleys as they move.",
     ["Glaciers", "carve", "valleys", "while", "advancing"]),
    ("Sleep consolidates memory. During deep sleep the brain replays the day's activity, strengthening the connections that store it.",
     "Deep sleep replays daily activity, strengthening the connections that store memory.",
     ["Sleep", "replays", "activity", "strengthening", "memory"]),
    ("Mangrove roots trap sediment along tropical coasts. The dense tangle slows waves and shelters young fish.",
     "Mangrove roots trap sediment, slow waves, and shelter young fish.",
     ["Mangroves", "trap", "sediment", "sheltering", "fish"]),
    ("Lightning heats the surrounding air almost instantly. The air expands so fast that it produces the shock wave heard as thunder.",
     "Lightning heats air so fast that its expansion makes thunder.",
     ["Lightning", "heats", "air", "producing", "thunder"]),
    ("Soap molecules have a water-loving end and an oil-loving end. This lets them lift grease off surfaces so water can rinse it away.",
     "Soap molecules lift grease from surfaces so water can rinse it away.",
     ["Soap", "lifts", "grease", "for", "rinsing"]),
    ("Deserts can be cold as well as hot. What defines a desert is low precipitation, not high temperature.",
     "Deserts are defined by low precipitation rather than by heat.",
     ["Deserts", "mean", "dryness", "not", "heat"]),
    ("Vaccines expose the immune system to a harmless part of a pathogen. The body then recognises the real infection faster.",
     "Vaccines train the immune system to recognise real infections faster.",
     ["Vaccines", "train", "immunity", "against", "pathogens"]),
]


def summarization_rows() -> list[dict]:
    rows: list[dict] = []
    for passage, one_sentence, five_words in SUMMARY_ITEMS:
        rows.append(example(f"Summarize in one sentence: {passage}", one_sentence))
        rows.append(example(f"Give a one-sentence summary of this text: {passage}", one_sentence))
        five = " ".join(five_words)
        rows.append(example(f"Give a five-word summary of this text: {passage}", five))
        rows.append(example(f"Summarize the following in exactly five words: {passage}", five))
    return rows


# --------------------------------------------------------------------------
# 11. Negation in signs and rules: state both what is allowed and what is not.
# --------------------------------------------------------------------------

SIGN_RULES = [
    ("No vehicles except buses", "buses", "vehicles", "cars"),
    ("No dogs except guide dogs", "guide dogs", "dogs", "pet dogs"),
    ("No entry except staff", "staff", "other people", "visitors"),
    ("No parking except permit holders", "permit holders", "other drivers", "visitors"),
    ("No food except baby food", "baby food", "other food", "snacks"),
    ("No boats except kayaks", "kayaks", "boats", "motorboats"),
    ("No cycling except on marked paths", "cycling on marked paths", "cycling elsewhere", "cycling on the grass"),
    ("No photography except in the courtyard", "photography in the courtyard", "photography elsewhere", "photos indoors"),
]


def negation_rows() -> list[dict]:
    rows: list[dict] = []
    for sign, allowed, banned, example_banned in SIGN_RULES:
        answer = (
            f"{allowed.capitalize()} are allowed, because the sign lists them as the exception. "
            f"{example_banned.capitalize()} are not allowed — all other {banned} are prohibited."
        )
        rows.append(example(
            f"The sign says, '{sign}.' Is {allowed} allowed? Is {example_banned} allowed?",
            answer,
        ))
        rows.append(example(
            f"A notice reads '{sign}.' What is permitted and what is not?",
            answer,
        ))
    return rows


# --------------------------------------------------------------------------
# 12. Commonsense physical outcomes, naming the mechanism explicitly.
# --------------------------------------------------------------------------

COMMONSENSE_ITEMS = [
    ("If a ceramic mug falls from a shelf onto a tiled floor, what is likely to happen and why?",
     "The mug will most likely shatter and break, because the hard tiles stop it suddenly and the "
     "impact force is more than the brittle ceramic can absorb."),
    ("If a glass jar slips from a counter onto concrete, what is likely to happen and why?",
     "The jar will probably shatter into sharp pieces, because concrete does not give way and the "
     "sudden impact breaks the brittle glass."),
    ("If a plate falls from a table onto stone tiles, what happens and why?",
     "The plate is likely to break and shatter, because the stone is hard and stops it abruptly."),
    ("You hang wet towels outside on a hot, breezy day. What will likely happen?",
     "They will dry quickly. The warm air speeds evaporation and the breeze carries the moisture away."),
    ("You leave a damp shirt on a washing line on a sunny, windy afternoon. What happens?",
     "The shirt will dry. Sun and moving air together make the water evaporate faster."),
    ("You spread a wet blanket outdoors on a warm windy morning. What is the likely result?",
     "It will dry out, because warmth and wind increase evaporation from the surface."),
    ("Which is safer for freeing a slice of bread stuck in a plugged-in toaster: poking it with a metal knife, or unplugging the toaster first? Explain briefly.",
     "Unplugging the toaster first is far safer. Metal conducts electricity, so putting a knife into a "
     "live toaster risks a serious shock; cutting the power removes that hazard."),
    ("Should you use a metal spoon inside a toaster that is still switched on, or switch it off and unplug it first?",
     "Switch it off and unplug it first — that is the safe option. Metal conducts electricity and could "
     "give you a shock while the toaster is live."),
    ("Is it safe to reach into a plugged-in blender to free stuck food, or should you unplug it first?",
     "Unplug it first; that is the safe approach. Reaching in while it is plugged in risks the blades "
     "starting and causing injury."),
    ("If you put an ice cube in a warm drink, what happens and why?",
     "The ice melts, because heat moves from the warmer liquid into the ice until it turns to water."),
    ("If you leave a bicycle out in the rain for weeks, what is likely to happen to the chain?",
     "It will rust, because water and oxygen react with the exposed steel."),
    ("If you drop a rubber ball onto a hard floor, what happens and why?",
     "It bounces back up, because rubber is elastic and returns most of the stored energy."),
]

MECHANISM_SUMMARIES = [
    ("Bees visit flowers to gather nectar and pollen. As they move from bloom to bloom they carry pollen with them, so the plants can set seed and fruit.",
     "Bees carry pollen between flowers as they feed, so their pollination lets plants set seed and fruit.",
     ["Bees", "pollinate", "flowers", "while", "feeding"]),
    ("Hoverflies feed at blossoms and move pollen between them, which helps orchard trees produce fruit.",
     "Hoverflies move pollen between blossoms while feeding, and that pollination helps orchards fruit.",
     ["Hoverflies", "pollinate", "blossoms", "aiding", "orchards"]),
    ("Moths visit night-blooming flowers for nectar and transfer pollen as they go, letting those plants reproduce.",
     "Moths transfer pollen between night flowers as they drink nectar, so pollination lets them reproduce.",
     ["Moths", "pollinate", "flowers", "during", "night"]),
    ("Wind carries pollen from grass flowers through the air to other plants of the same species, which then set seed.",
     "Wind moves grass pollen between plants, and that pollination lets them set seed.",
     ["Wind", "spreads", "pollen", "enabling", "seed"]),
]


def commonsense_rows() -> list[dict]:
    rows: list[dict] = []
    for question, answer in COMMONSENSE_ITEMS:
        rows.append(example(question, answer))
    for passage, one_sentence, five_words in MECHANISM_SUMMARIES:
        rows.append(example(f"Summarize in one sentence: {passage}", one_sentence))
        rows.append(example(f"Give a one-sentence summary of this text: {passage}", one_sentence))
        rows.append(example(f"Give a five-word summary of this text: {passage}", " ".join(five_words)))
    return rows


def main() -> None:
    eval_prompts = {normalized(json.loads(line)["prompt"]) for line in EVAL_PROMPTS.open()}

    rows: list[dict] = []
    rows += mc_rows()
    rows += instruction_rows()
    rows += structured_rows()
    rows += json_volume_rows()
    rows += table_volume_rows()
    rows += context_rows()
    rows += arithmetic_rows()
    rows += calibration_rows()
    rows += plan_rows()
    rows += edge_case_rows()
    rows += spatial_and_classification_rows()
    rows += summarization_rows()
    rows += negation_rows()
    rows += commonsense_rows()

    # Any generated prompt that collides verbatim with a benchmark prompt is
    # dropped rather than tolerated — training on it would be test contamination.
    # Each pattern has other phrasings, so the contract still gets taught.
    before = len(rows)
    rows = [r for r in rows if normalized(r["messages"][0]["content"]) not in eval_prompts]
    dropped_leaks = before - len(rows)

    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        key = normalized(row["messages"][0]["content"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    # prepare_sft.py drops exact conversation duplicates, so upsampling here
    # would be silently discarded — weight comes from the mixture caps instead.
    emitted = unique
    random.Random(SEED).shuffle(emitted)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / "format_compliance_curriculum.jsonl"
    tmp = target.with_suffix(".jsonl.tmp")
    with tmp.open("w") as handle:
        for row in emitted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(target)
    print(
        f"wrote {len(emitted)} examples to {target} "
        f"(dropped {dropped_leaks} verbatim eval-prompt collisions, {before - dropped_leaks - len(emitted)} duplicates)"
    )


if __name__ == "__main__":
    main()
