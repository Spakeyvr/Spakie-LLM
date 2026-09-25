"""Deterministic number-magnitude curriculum: pairwise and entity comparisons.

Motivation: the 94M base scores at chance on forced-choice pairwise comparison
of 3-digit numbers, so answer-only three-way comparison data cannot build on an
existing skill. This curriculum teaches the pairwise primitive (answer-first and
worked digit-by-digit) and composes it into named three-way comparisons.

Evaluation templates are kept out: no document uses "Compare the numbers" or
"The larger/smaller number is", and number pairs / entity names from supplied
evaluation task files are excluded. Every document's stated answer is checked
against an oracle before writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = ['Arlo', 'Bexa', 'Cato', 'Dela', 'Esko', 'Fenn', 'Gisa', 'Hollo', 'Imra', 'Joss', 'Kavi', 'Lomi',
         'Mako', 'Nesa', 'Orin', 'Pell', 'Quin', 'Rasa', 'Sivo', 'Tamsi', 'Ulla', 'Veno', 'Wren', 'Xavo',
         'Yara', 'Zeph', 'Brisa', 'Colm', 'Dasha', 'Eril', 'Fiona', 'Galt', 'Hesper', 'Ivo', 'Jaro', 'Kesh']
THINGS = [('scored', 'points', 'the most points', 'the fewest points'),
          ('collected', 'shells', 'the most shells', 'the fewest shells'),
          ('ran', 'meters', 'the longest distance', 'the shortest distance'),
          ('saved', 'coins', 'the most coins', 'the fewest coins'),
          ('planted', 'seeds', 'the most seeds', 'the fewest seeds')]
BANNED = ('Compare the numbers', 'The larger number is', 'The smaller number is')
PLACES = ('hundreds', 'tens', 'units')


def digits(n):
    return [n // 100, n // 10 % 10, n % 10]


def deciding_place(a, b):
    """Index of the first differing digit of two distinct 3-digit numbers."""
    if a == b:
        raise ValueError('Equal numbers have no deciding place')
    return next(i for i, (x, y) in enumerate(zip(digits(a), digits(b))) if x != y)


def sample_pair(rng, place, banned_pairs):
    while True:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        if a != b and deciding_place(a, b) == place and frozenset((a, b)) not in banned_pairs:
            return a, b


def worked_pair(a, b, symmetric=False):
    da, db = digits(a), digits(b)
    lines = [f'Digit check for {a} and {b}.']
    for place, x, y in zip(PLACES, da, db):
        if x == y:
            lines.append(f'The {place} digits are both {x}.')
        else:
            big, small = (x, y) if x > y else (y, x)
            lines.append(f'The {place} digits are {x} and {y}; {big} is more than {small}, so this place decides.')
            break
    ending = f'So {max(a, b)} is greater than {min(a, b)}'
    lines.append(ending + (f', and {min(a, b)} is less than {max(a, b)}.' if symmetric else '.'))
    return '\n'.join(lines)


LARGER = (0, 2, 4, 6)
SMALLER = (1, 3, 5)


def answer_first_pair(rng, a, b, direction=None):
    """direction None: any template; 'larger'/'smaller': templates asking for that side only."""
    hi, lo = max(a, b), min(a, b)
    templates = [
        f'Of {a} and {b}, the greater one is {hi}.',
        f'Of {a} and {b}, the lesser one is {lo}.',
        f'Which is bigger, {a} or {b}? {hi}.',
        f'Which is less, {a} or {b}? {lo}.',
        f'{hi} is greater than {lo}.',
        f'{lo} is less than {hi}.',
        f'Between {a} and {b}, {hi} is the maximum and {lo} is the minimum.',
    ]
    if direction is None:
        return rng.choice(templates)
    return templates[rng.choice(LARGER if direction == 'larger' else SMALLER)]


RICH_WORDS = {'larger': ('larger', 'bigger', 'greater', 'higher'), 'smaller': ('smaller', 'lesser', 'lower', 'less')}


def rich_pair(rng, a, b, direction):
    """Answer-first pair using a varied direction word; never the evaluation template."""
    word = rng.choice(RICH_WORDS[direction])
    answer = max(a, b) if direction == 'larger' else min(a, b)
    return rng.choice([
        f'Of {a} and {b}, the {word} value is {answer}.',
        f'Which number is {word}: {a} or {b}? It is {answer}.',
        f'Pick the {word} of {a} and {b}: {answer}.',
        f'{a} or {b}? The {word} one is {answer}.',
    ])


def three_way(rng, values, banned_names):
    names = rng.sample([n for n in NAMES if n not in banned_names], 3)
    verb, unit, most, fewest = rng.choice(THINGS)
    facts = ' '.join(f'{n} {verb} {v} {unit}.' for n, v in zip(names, values))
    hi_name, lo_name = names[values.index(max(values))], names[values.index(min(values))]
    if rng.random() < .5:
        ordered = ', '.join(str(v) for v in sorted(values, reverse=True))
        body = (f'{facts}\nFrom largest to smallest the numbers are {ordered}.\n'
                f'So {hi_name} has {most} and {lo_name} has {fewest}.')
    else:
        want_most = rng.random() < .5
        body = f"{facts}\nThe one with {most if want_most else fewest} is {hi_name if want_most else lo_name}."
    return body, {'names': names, 'values': values, 'max_name': hi_name, 'min_name': lo_name}


THREE_WAY_ITEMS = [('scored', 'points'), ('collected', 'shells'), ('saved', 'coins'), ('planted', 'seeds'),
                   ('sold', 'tickets'), ('read', 'pages'), ('baked', 'rolls'), ('counted', 'stars')]
SEPARATORS = ('. ', '; ', ', ')
HIGH = ('the most', 'the highest number of', 'the largest number of', 'the greatest number of')
LOW = ('the fewest', 'the lowest number of', 'the smallest number of', 'the least number of')
ASKERS = ('The person with {q} {unit} is', 'Who has {q} {unit}? It is', 'The one who {verb} {q} {unit} is')


def three_way_answer_first(rng, values, banned_names):
    """Direct-answer three-way comparison with varied wording; answer position is uniform."""
    names = rng.sample([n for n in NAMES if n not in banned_names], 3)
    verb, unit = rng.choice(THREE_WAY_ITEMS)
    sep = rng.choice(SEPARATORS)
    listing = sep.join(f'{n} {verb} {v} {unit}' for n, v in zip(names, values)) + '.'
    want_high = rng.random() < .5
    q = rng.choice(HIGH if want_high else LOW)
    answer = names[values.index(max(values) if want_high else min(values))]
    ask = rng.choice(ASKERS).format(q=q, unit=unit, verb=verb)
    return f'{listing}\n{ask} {answer}.', {'names': names, 'values': values, 'answer': answer, 'want_high': want_high}


def verify(doc):
    """Oracle check: every 'X is greater/less than Y' and ordering claim must be true."""
    text = doc['text']
    if any(b in text for b in BANNED):
        raise ValueError(f'Evaluation template leaked: {text!r}')
    for x, rel, y in re.findall(r'(\d{3}) is (greater|less) than (\d{3})', text):
        if (int(x) > int(y)) != (rel == 'greater'):
            raise ValueError(f'False comparison claim: {text!r}')
    for x, y, z in re.findall(r'largest to smallest the numbers are (\d{3}), (\d{3}), (\d{3})', text):
        if not int(x) > int(y) > int(z):
            raise ValueError(f'False ordering: {text!r}')
    meta = doc.get('meta')
    if meta and 'answer' in meta:
        vals = meta['values']
        expected = meta['names'][vals.index(max(vals) if meta['want_high'] else min(vals))]
        if meta['answer'] != expected or not text.endswith(f' {expected}.'):
            raise ValueError(f'Wrong three-way answer: {text!r}')
    elif meta and 'names' in meta:
        vals = meta['values']
        assert meta['max_name'] == meta['names'][vals.index(max(vals))]
        assert meta['min_name'] == meta['names'][vals.index(min(vals))]
        for name in meta['names']:
            if re.search(rf'\b{name}\b has the (most|longest)', text) and name != meta['max_name']:
                raise ValueError(f'Wrong maximum entity: {text!r}')
    return True


def banned_from_tasks(paths):
    pairs, names = set(), set()
    for path in paths:
        for task in json.loads(Path(path).read_text()):
            values = [int(v) for v in task.get('values', [])]
            pairs.update(frozenset(p) for p in combinations(values, 2))
            names.update(str(e) for e in task.get('entities', []))
    return pairs, names


def build(n_documents, seed, banned_pairs=frozenset(), banned_names=frozenset(), symmetric=False, rich=False,
          three_way_only=False):
    rng = random.Random(seed)
    docs = []
    for i in range(n_documents):
        if three_way_only:
            while True:
                values = rng.sample(range(100, 1000), 3)
                if all(frozenset(p) not in banned_pairs for p in combinations(values, 2)):
                    break
            text, meta = three_way_answer_first(rng, values, banned_names)
            doc = {'kind': 'three_way_answer_first', 'text': text, 'meta': meta}
            verify(doc)
            docs.append(doc)
            continue
        kind = ('answer_first', 'worked', 'three_way')[i % 3]
        place = (i // 3) % 3
        if kind == 'three_way':
            while True:
                values = rng.sample(range(100, 1000), 3)
                if all(frozenset(p) not in banned_pairs for p in combinations(values, 2)):
                    break
            text, meta = three_way(rng, values, banned_names)
        else:
            a, b = sample_pair(rng, place, banned_pairs)
            if kind == 'answer_first':
                direction = ('larger', 'smaller')[(i // 9) % 2] if symmetric or rich else None
                text = rich_pair(rng, a, b, direction) if rich else answer_first_pair(rng, a, b, direction)
            else:
                text = worked_pair(a, b, symmetric)
            meta = {'pair': [a, b], 'deciding_place': PLACES[place]}
        doc = {'kind': kind, 'text': text, 'meta': meta}
        verify(doc)
        docs.append(doc)
    return docs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for curriculum.jsonl/.npy/manifest.json')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--documents', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=20260923)
    parser.add_argument('--exclude-tasks', type=Path, nargs='*', default=[])
    parser.add_argument('--symmetric', action='store_true',
                        help='Balance larger/smaller answer-first items and state both directions in worked items')
    parser.add_argument('--three-way-only', action='store_true',
                        help='Only direct-answer three-way comparisons with varied wording')
    parser.add_argument('--rich', action='store_true',
                        help='Balanced answer-first items with varied direction words (smaller/larger/lower/higher...)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    import numpy as np
    from tokenizer.train_tokenizer import SpakieTokenizer
    banned_pairs, banned_names = banned_from_tasks(args.exclude_tasks)
    docs = build(args.documents, args.seed, banned_pairs, banned_names, symmetric=args.symmetric, rich=args.rich,
                 three_way_only=args.three_way_only)
    tokenizer = SpakieTokenizer(str(args.tokenizer))
    ids = [tokenizer.encode(d['text'], add_eos=True) for d in docs]
    args.output.mkdir(parents=True)
    with (args.output / 'curriculum.jsonl').open('w') as stream:
        for d in docs:
            stream.write(json.dumps(d) + '\n')
    array = np.concatenate([np.asarray(x, dtype=np.int32) for x in ids])
    np.save(args.output / 'curriculum.npy', array)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    counts = {}
    for d in docs:
        counts[d['kind']] = counts.get(d['kind'], 0) + 1
    manifest = {'symmetric': args.symmetric, 'rich': args.rich, 'three_way_only': args.three_way_only, 'documents': len(docs), 'tokens': int(len(array)), 'counts': counts, 'seed': args.seed,
                'excluded_pairs': len(banned_pairs), 'excluded_names': sorted(banned_names),
                'exclude_tasks_sha256': {str(p): digest(p) for p in args.exclude_tasks},
                'tokenizer_sha256': digest(args.tokenizer), 'script_sha256': digest(__file__),
                'curriculum_sha256': digest(args.output / 'curriculum.npy'),
                'documents_sha256': digest(args.output / 'curriculum.jsonl'), 'all_documents_verified': True}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'excluded_names'}))


if __name__ == '__main__':
    main()
