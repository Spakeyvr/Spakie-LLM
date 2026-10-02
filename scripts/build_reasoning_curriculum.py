"""Verified plain-text context/reasoning data for bounded BASE continuation probes.

Context training and evaluation separate entities, values, and templates.
Arithmetic training excludes held-out operand pairs, including their occurrence
as intermediate training equations; individual numbers may be shared.
These synthetic diagnostics measure limited transfer, not general intelligence.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_base_learning import build_tasks, digest, pair_key
from tokenizer.train_tokenizer import SpakieTokenizer


TRAIN_NAMES = ('Alice', 'Bob', 'Carmen', 'David', 'Elena', 'Farah', 'George',
               'Hana', 'Isaac', 'Julia', 'Kevin', 'Leah', 'Mina', 'Noah',
               'Olivia', 'Peter', 'Quinn', 'Rosa', 'Sofia', 'Tomas')
DEV_NAMES = ('Adri', 'Bex', 'Cavo', 'Deni', 'Esha', 'Favi', 'Garo', 'Hila')
TEST_NAMES = ('Ivo', 'Jexa', 'Kavu', 'Lemi', 'Mora', 'Navo', 'Osha', 'Peli')
ATTRIBUTES = ('locker number', 'ticket number', 'room number', 'access code')
REASONING_SCORING_VERSION = 6


def score_reasoned_completion(text, row, *, budget_reached=False):
    """Score the explicit final answer and all three place-value calculations.

    The final answer is the first claim after "Answer:", judged like every BASE
    numeric probe (see scripts/_base_probe_cases.numeric_claim).
    """
    from scripts._base_probe_cases import NUMERIC_LITERAL, numeric_claim
    match = re.search(r'\bAnswer:\s*', text, re.I)
    answer_claim = numeric_claim(text[match.end():], budget_reached=budget_reached) if match else None
    answer_correct = bool(answer_claim and answer_claim['status'] == 'claimed'
                          and Decimal(answer_claim['value']) == Decimal(str(row['answer'])))
    prefix = text[:match.start()] if match else text
    # Parse full signed numbers: a wrong decimal or negative value must not
    # masquerade as a correct integer intermediate result.
    number = '(' + NUMERIC_LITERAL + ')'
    pattern = r'(?<![\w.,])' + number + r'\s*\+\s*' + number + r'\s*=\s*' + number + r'(?!\w|[.,]\d)'
    equations = [tuple(Decimal(v.replace(',', '')) for v in m) for m in re.findall(pattern, prefix)]
    def canonical(equation):
        a, b, result = equation
        return min(a,b), max(a,b), result
    steps_correct = answer_correct and sorted(map(canonical, equations)) == sorted(
        map(canonical, row['expected_equations']))
    return {'answer_correct': answer_correct, 'steps_correct': steps_correct}



def addition_claim_diagnostics(text):
    """Audit explicit numeric addition claims anywhere in a generated answer.

    Recognizes signed integers/decimals (including grouped thousands) using
    either `+`/`=` or `plus`/`equals`. This is a supplementary diagnostic, not
    certification of every statement: algebra, scientific notation, number
    words and other operations are outside this parser's scope.
    """
    from scripts._base_probe_cases import NUMERIC_LITERAL
    number = '(' + NUMERIC_LITERAL + ')'
    pattern = (r'(?<![\w.,+-])' + number + r'(?:\s*\+\s*|\s+plus\s+)'
               + number + r'(?:\s*=\s*|\s+equals\s+)' + number
               + r'(?!\w|[.,]\d)')
    # A result may also start the next claim: "1+2=3+4=9". A consuming
    # search skips the false second addition, so inspect overlapping starts.
    matches = list(re.finditer('(?=(' + pattern + '))', text, re.IGNORECASE))
    invalid = []
    for match in matches:
        literals = match.groups()[1:]
        values = [Decimal(value.replace(',', '')) for value in literals]
        # Avoid Decimal's default 28-digit rounding on long model outputs.
        with localcontext() as context:
            context.prec = max(28, sum(len(value) for value in literals) + 2)
            correct = values[0] + values[1] == values[2]
        if not correct:
            invalid.append({'claim': match.group(1), 'start': match.start(1), 'end': match.end(1)})
    return {'recognized_addition_claims': len(matches), 'invalid_addition_claims': invalid}


def example(rng, index, split='train', *, train_values=None):
    if split not in ('train', 'dev', 'test'):
        raise ValueError('Unknown split')
    names = TRAIN_NAMES if split == 'train' else DEV_NAMES if split == 'dev' else TEST_NAMES
    entities = [str(x) for x in rng.choice(names, size=3, replace=False)]
    # Entire value ranges differ across splits, not just sampled records.
    low, high = (100, 400) if split == 'train' else (500, 700) if split == 'dev' else (700, 900)
    values = [int(x) for x in rng.choice(train_values if split == 'train' and train_values is not None
                                        else np.arange(low, high), size=3, replace=False)]
    selected = int(rng.integers(3))
    attribute = ATTRIBUTES[index % len(ATTRIBUTES)]
    kind = ('retrieve', 'maximum', 'minimum')[index % 3]
    if kind == 'retrieve':
        expected = str(values[selected])
        if split == 'train':
            form = index // 3 % 4
            statements = [f'{n} has {v} as their {attribute}.' for n, v in zip(entities, values)]
            if form == 0:
                prompt = ' '.join(statements) + f'\nQuestion: What is the {attribute} for {entities[selected]}?\nAnswer:'
            elif form == 1:
                prompt = 'Records:\n' + '\n'.join(f'{n}: {attribute} {v}' for n, v in zip(entities, values))
                prompt += f'\nThe {attribute} belonging to {entities[selected]} is'
            elif form == 2:
                prompt = ' '.join(statements) + f'\nLooking up {entities[selected]}, we find the {attribute}'
            else:
                prompt = ' '.join(statements) + f'\nFind {entities[selected]} in the records. Their {attribute} is'
        elif split == 'dev':
            prompt = 'Directory: ' + '; '.join(f'{n} — {attribute}: {v}' for n, v in zip(entities, values))
            prompt += f'.\nWhich {attribute} belongs to {entities[selected]}?\nAnswer:'
        else:
            prompt = 'An administrator recorded ' + ', '.join(f'{v} for {n}' for n, v in zip(entities, values))
            prompt += f' as the respective {attribute}s.\nFor {entities[selected]}, the recorded number was'
    else:
        selected = values.index(max(values) if kind == 'maximum' else min(values))
        expected = entities[selected]
        if split == 'train':
            statements = ' '.join(f'{n} collected {v} points.' for n, v in zip(entities, values))
            word = 'most' if kind == 'maximum' else 'fewest'
            if index // 3 % 2:
                prompt = statements + f'\nQuestion: Who collected the {word} points?\nAnswer:'
            else:
                prompt = statements + f'\nThe person with the {word} points is'
        elif split == 'dev':
            prompt = 'Scores: ' + ', '.join(f'{n} = {v}' for n, v in zip(entities, values))
            word = 'highest' if kind == 'maximum' else 'lowest'
            prompt += f'.\nName the person whose score is {word}.\nAnswer:'
        else:
            prompt = 'The results were ' + '; '.join(f'{n} scored {v}' for n, v in zip(entities, values))
            word = 'largest' if kind == 'maximum' else 'smallest'
            prompt += f'.\nThe {word} score was achieved by'
    return {'kind': kind, 'entities': entities, 'values': values, 'selected': selected,
            'prompt': prompt, 'answer': expected, 'text': prompt + ' ' + expected + '.',
            'template_split': split}


def addition_explanations(count=8192, seed=20260924):
    """Expose verifiable place-value steps; exclude held-out intermediate pairs too."""
    from scripts.eval_base_readiness import cases
    from scripts._base_probe_cases import suite
    from scripts.eval_reasoning_probe import SEALED_PAIRS
    _, old_tasks = build_tasks(worked_examples=True)
    excluded = {tuple(r['pair']) for r in old_tasks if 'pair' in r}
    excluded.update(pair_key('+', a, b) for a, b in SEALED_PAIRS)
    for split in ('dev', 'test'):
        excluded.update(pair_key('+', *r['operands']) for r in cases(split) if 'operands' in r)
    for row in suite():
        for a, op, b in re.findall(r'(\d+)\s*([+*/-])\s*(\d+)\s*=', row['prompt']):
            excluded.add(pair_key(op, int(a), int(b)))
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        a, b = (int(x) for x in rng.integers(1, 200, size=2))
        at, bt, au, bu = a//10*10, b//10*10, a%10, b%10
        steps = [(at, bt), (au, bu), (at+bt, au+bu)]
        pairs = [pair_key('+', a, b)] + [pair_key('+', x, y) for x, y in steps]
        if any(p in excluded for p in pairs):
            continue
        i = len(rows)
        problem = [f'Add {a} and {b}.', f'Compute {a} + {b}.',
                   f'A store had {a} pencils and obtained {b} more. How many pencils does it have now?',
                   f'There are {a} adults and {b} children on a train. How many people are on the train?'][i % 4]
        text = (f'Problem: {problem}\nSolution: Separate tens and units. '
                f'Tens: {at} + {bt} = {at+bt}. Units: {au} + {bu} = {au+bu}. '
                f'Combine: {at+bt} + {au+bu} = {a+b}.\nAnswer: {a+b}.')
        # Direct forms bridge computed results back to ordinary completions.
        text += f'\n{a} + {b} = {a+b}. {a} plus {b} equals {a+b}.'
        rows.append({'kind':'addition_explanation', 'text':text,
                     'a':a, 'b':b, 'answer':str(a+b), 'pairs':[list(p) for p in pairs]})
    return rows


def build(seed=20260923, count=16384, *, decomposed_addition=False, broad_numbers=False,
          retrieval_only=False):
    rng = np.random.default_rng(seed)
    arithmetic, old_tasks = build_tasks(worked_examples=True)
    train_values = None
    if broad_numbers:
        excluded_values = set()
        for split, offset in [('dev',1), ('test',2)]:
            erng = np.random.default_rng(seed + offset)
            for i in range(96):
                excluded_values.update(example(erng, i, split)['values'])
        train_values = np.array([v for v in range(100,1000) if v not in excluded_values])
    documents = [example(rng, i*3 if retrieval_only else i, train_values=train_values)
                 for i in range(count)]
    # Preserve broad, verified arithmetic/code exposure without a chat template.
    documents.extend(arithmetic)
    if decomposed_addition:
        documents.extend(addition_explanations())
    rng.shuffle(documents)
    splits = {}
    for split, offset in [('dev', 1), ('test', 2)]:
        erng = np.random.default_rng(seed + offset)
        tasks = []
        for i in range(96):
            row = example(erng, i, split)
            tasks.append({'id': f'{split}_{i}', 'category': 'context_' + row['kind'],
                          'prompt': row['prompt'], 'expected_regex': re.escape(row['answer']) + r'\b',
                          'answer': row['answer'], 'max_new_tokens': 12,
                          'entities': row['entities'], 'values': row['values'],
                          'selected': row['selected'], 'template_split': split})
        splits[split] = tasks
    return documents, old_tasks, splits


def prepare(tokenizer_path, output, count=16384, *, decomposed_addition=False, broad_numbers=False,
            retrieval_only=False):
    if output.exists():
        raise ValueError('Output already exists; use a new directory')
    tokenizer = SpakieTokenizer(str(tokenizer_path))
    documents, old_tasks, splits = build(count=count, decomposed_addition=decomposed_addition,
                                       broad_numbers=broad_numbers, retrieval_only=retrieval_only)
    output.mkdir(parents=True)
    tokens = []
    with (output/'curriculum.jsonl').open('w') as stream:
        for row in documents:
            stream.write(json.dumps(row) + '\n')
            tokens.extend(tokenizer.encode(row['text'], add_eos=True))
    np.save(output/'curriculum.npy', np.asarray(tokens, dtype=np.uint16))
    (output/'tasks.json').write_text(json.dumps(old_tasks + splits['dev'], indent=2) + '\n')
    (output/'sealed_tasks.json').write_text(json.dumps(splits['test'], indent=2) + '\n')
    manifest = {'tokenizer_sha256': digest(tokenizer_path),
                'curriculum_sha256': digest(output/'curriculum.npy'),
                'tasks_sha256': digest(output/'tasks.json'),
                'sealed_tasks_sha256': digest(output/'sealed_tasks.json'),
                'documents': len(documents), 'tokens': len(tokens), 'seed': 20260923,
                'generator_sha256': digest(__file__),
                'decomposed_addition': decomposed_addition,
                'broad_numbers': broad_numbers,
                'retrieval_only': retrieval_only,
                'objective': 'All-token BASE prediction, no chat roles or assistant masks.',
                'scope': 'Synthetic retrieval/comparison plus arithmetic/code; distinct held-out entities, numeric values and templates.'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decomposed-addition', action='store_true',
                        help='Include verified place-value explanations with held-out operand pairs excluded')
    parser.add_argument('--broad-numbers', action='store_true',
                        help='Cover all three-digit leading digits while excluding fixed dev/test values')
    parser.add_argument('--retrieval-only', action='store_true',
                        help='Use every context document for retrieval; omit the unproven min/max curriculum')
    args = parser.parse_args()
    prepare(args.tokenizer, args.output, decomposed_addition=args.decomposed_addition,
            broad_numbers=args.broad_numbers, retrieval_only=args.retrieval_only)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped; partial preparation must use a new output directory.', file=sys.stderr)
        raise SystemExit(130)
