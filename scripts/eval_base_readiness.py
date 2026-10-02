"""Frozen BASE completion checks, independent of the learning-probe curriculum.

These small diagnostics expose failure modes; they are not public benchmark
scores. Development and test facts, operands, names, and grammar pairs differ.
No chat template or assistant loss mask is used.

Answer rows get two scores: pass/fail on the first claim of a greedy completion,
whatever its phrasing (see scripts/_base_probe_cases.py), and a likelihood rank
of the correct answer against plausible rivals. Format instructions are scored
exactly; grammar compares sentence likelihoods.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cases(split='dev'):
    if split not in ('dev', 'test'):
        raise ValueError('Unknown split')
    test = split == 'test'
    rows = []
    # Rivals are the country's other well-known cities and nearby capitals: the
    # answers a confused model plausibly gives instead.
    facts = ([('Spain', 'Madrid', ['Barcelona', 'Seville', 'Valencia']),
              ('Portugal', 'Lisbon', ['Porto', 'Madrid', 'Coimbra']),
              ('Norway', 'Oslo', ['Bergen', 'Stockholm', 'Trondheim']),
              ('Sweden', 'Stockholm', ['Gothenburg', 'Oslo', 'Malmö']),
              ('Finland', 'Helsinki', ['Tampere', 'Turku', 'Stockholm']),
              ('Greece', 'Athens', ['Thessaloniki', 'Sparta', 'Rome']),
              ('Austria', 'Vienna', ['Salzburg', 'Graz', 'Berlin']),
              ('Poland', 'Warsaw', ['Krakow', 'Gdansk', 'Prague'])]
             if not test else
             [('Canada', 'Ottawa', ['Toronto', 'Montreal', 'Vancouver']),
              ('Egypt', 'Cairo', ['Alexandria', 'Giza', 'Luxor']),
              ('Kenya', 'Nairobi', ['Mombasa', 'Kisumu', 'Addis Ababa']),
              ('Ireland', 'Dublin', ['Cork', 'Belfast', 'Galway']),
              ('Denmark', 'Copenhagen', ['Aarhus', 'Odense', 'Oslo']),
              ('Hungary', 'Budapest', ['Debrecen', 'Vienna', 'Prague']),
              ('Thailand', 'Bangkok', ['Chiang Mai', 'Phuket', 'Hanoi']),
              ('Argentina', 'Buenos Aires', ['Córdoba', 'Rosario', 'Santiago'])])
    for country, capital, rivals in facts:
        rows.append(dict(category='facts', prompt=f'The capital of {country} is', answer=capital,
                         rivals=rivals, max_new_tokens=16))
    # Both surface forms use the same operands to expose format transfer.
    for i in range(8):
        a, b = (31 + i * 7 + 100 * test), (12 + i * 3 + 20 * test)
        for style in ('equation', 'word_problem'):
            prompt = (f'12 + 5 = 17\n{a} + {b} =' if style == 'equation' else
                      f'A box contains {a} red beads and {b} blue beads. '
                      'The total number of beads in the box is')
            rows.append(dict(category=style, prompt=prompt, answer=str(a+b),
                             max_new_tokens=16, operands=[a,b]))
    names = ['Tarin','Luro','Nemi','Pavo','Risa','Velo','Zuna','Kori'] if not test else [
             'Davo','Fera','Humi','Jora','Meko','Savi','Tulo','Weri']
    for i, name in enumerate(names):
        code = str(431 + 17 * i + 200 * test)
        rows.append(dict(category='grounding', prompt=f'The badge code for {name} is {code}. '
                         f'The badge code for Bela is 902.\nQuestion: What is the badge code for {name}?\nAnswer:',
                         # Room for a sentence-form answer plus its terminator.
                         answer=code, max_new_tokens=20))
        rows.append(dict(category='format_instruction',
                         prompt=f'Write only the word {name.upper()}. Do not add anything else.\nAnswer:',
                         answer=name.upper(), scoring='exact', max_new_tokens=12))
    pairs = ([('The children are playing outside.', 'The children is playing outside.'),
              ('She has finished her homework.', 'She have finished her homework.'),
              ('I saw an elephant.', 'I saw a elephant.'),
              ('He went to the market yesterday.', 'He go to the market yesterday.'),
              ('These apples taste sweet.', 'These apples tastes sweet.'),
              ('There are two chairs.', 'There is two chairs.'),
              ('They do not know the answer.', 'They does not know the answer.'),
              ('My sister likes music.', 'My sister like music.')]
             if not test else
             [('The dogs are barking loudly.', 'The dogs is barking loudly.'),
              ('He has washed his hands.', 'He have washed his hands.'),
              ('We found an empty room.', 'We found a empty room.'),
              ('She walked home last night.', 'She walk home last night.'),
              ('Those flowers smell nice.', 'Those flowers smells nice.'),
              ('There are three windows.', 'There is three windows.'),
              ('We do not need more time.', 'We does not need more time.'),
              ('The teacher reads books.', 'The teacher read books.')])
    for good, bad in pairs:
        rows.append(dict(category='grammar_preference', good=good, bad=bad))
    for prompt in (['A bicycle has two wheels. To ride it safely,',
                    'The young fox followed the path through the forest until'] if not test else
                   ['When the snow began to melt, the villagers',
                    'An electric motor converts electrical energy into']):
        rows.append(dict(category='continuation', prompt=prompt, max_new_tokens=64))
    for i, row in enumerate(rows):
        row['id'] = f'{split}_{i:03d}'
    return rows


def score_answer(text, row, *, budget_reached=None):
    """Pass/fail for an answer row: exact for format instructions, else the first claim."""
    if row.get('scoring') == 'exact':
        return text.strip() == row['answer']
    from scripts._base_probe_cases import score_completion
    return score_completion(text, row, budget_reached=budget_reached)


def evaluate_model(model, tokenizer_path, output, *, split='dev', stopped=lambda: False):
    import mlx.core as mx
    from inference.generate_mlx import generate
    from tokenizer.train_tokenizer import SpakieTokenizer
    from scripts._base_probe_cases import claim, repetition, SCORING_VERSION
    from scripts._base_probe_likelihood import likelihood_record
    from scripts.probe_pretrain_recipe import sha, write_json
    tok = SpakieTokenizer(str(tokenizer_path))
    tasks = cases(split)
    counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    likelihood = defaultdict(lambda: {'correct': 0, 'total': 0, 'margin_sum': 0.})
    result = {'split': split, 'scoring_version': SCORING_VERSION, 'task_sha256': hashlib.sha256(
        json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        'tokenizer_sha256': sha(tokenizer_path), 'records': [], 'metrics': {},
        'scope': 'Small diagnostic suite. Answer rows: first-claim pass/fail plus likelihood rank '
                 'against rivals. Grammar is length-normalized sentence likelihood; format '
                 'instructions are BASE completions, not a chat/SFT readiness claim.'}
    def nll(text):
        ids = tok.encode(text)
        _, loss, _ = model(mx.array([ids[:-1]]), mx.array([ids[1:]]), ignore_index=None)
        return float(loss.item())
    was_training = model.training
    model.eval()
    try:
        for row in tasks:
            if stopped():
                break
            record = dict(row)
            if row['category'] == 'grammar_preference':
                good, bad = nll(row['good']), nll(row['bad'])
                record.update(good_nll=good, bad_nll=bad, passed=good < bad)
            else:
                ids = generate(model, tok, tok.encode(row['prompt']), max_new_tokens=row['max_new_tokens'],
                               temperature=0., top_k=0, top_p=1., repetition_penalty=1.)
                record.update(completion=tok.decode(ids), repeated_4gram_fraction=repetition(ids))
                record.update(generated_tokens=len(ids),
                              generation_budget_reached=len(ids) >= row['max_new_tokens'])
                if 'answer' in row:
                    budget = record['generation_budget_reached']
                    if row.get('scoring') != 'exact':
                        record['claim'] = claim(record['completion'], row, budget_reached=budget)
                    record['passed'] = score_answer(record['completion'], row, budget_reached=budget)
                    scored = likelihood_record(model, tok, row) if row.get('scoring') != 'exact' else None
                    if scored:
                        record.update(scored)
                        stats = likelihood[row['category']]
                        stats['correct'] += int(scored['likelihood_correct'])
                        stats['total'] += 1
                        stats['margin_sum'] += scored['likelihood_margin']
            if 'passed' in record:
                counts[row['category']]['correct'] += int(record['passed'])
                counts[row['category']]['total'] += 1
            result['records'].append(record)
            result['metrics'] = {k: dict(v) for k, v in counts.items()}
            for k, v in likelihood.items():
                result['metrics'][k]['likelihood'] = {'correct': v['correct'], 'total': v['total'],
                                                      'mean_margin': v['margin_sum'] / v['total']}
            result['status'] = 'running'
            write_json(output, result)
        result['status'] = 'complete' if len(result['records']) == len(tasks) else 'interrupted'
        write_json(output, result)
    finally:
        model.train(was_training)
    return {k:v for k,v in result.items() if k != 'records'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['dev','test'], default='dev')
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Output already exists')
    from runtime.checkpoint_io import (load_mlx_checkpoint_config, load_mlx_checkpoint_meta,
        load_mlx_model_weights_strict, validate_checkpoint_tokenizer)
    from runtime.mlx_backend import load_safetensors
    from model.transformer_mlx import SpakieGPTMLX
    from scripts.probe_pretrain_recipe import sha, write_json
    metadata = load_mlx_checkpoint_meta(str(args.checkpoint))
    from scripts._base_probe_cases import is_base_checkpoint
    if not is_base_checkpoint(metadata, args.checkpoint):
        p.error('Use an explicitly identified BASE/pretrain checkpoint')
    validate_checkpoint_tokenizer(metadata, str(args.tokenizer), source=str(args.checkpoint))
    cfg = load_mlx_checkpoint_config(str(args.checkpoint))
    model = SpakieGPTMLX(cfg)
    load_mlx_model_weights_strict(model, load_safetensors(str(args.checkpoint)), path=str(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = False
    def interrupt(_sig, _frame):
        nonlocal stop
        stop = True
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        result = evaluate_model(model, args.tokenizer, args.output, split=args.split, stopped=lambda: stop)
        full = json.loads(args.output.read_text())
        full.update(checkpoint_sha256=sha(args.checkpoint), script_sha256=sha(__file__))
        write_json(args.output, full)
        print(json.dumps(result), flush=True)
        return 130 if stop else 0
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
