"""Forced-choice likelihood scoring for in-context retrieval and comparison tasks.

Greedy generation only shows whether the top answer is exactly right. This
diagnostic asks a weaker question: among the entities (or values) listed in the
prompt, does the model assign the highest total log-probability to the correct
one? Chance is 1/len(choices). It separates "partially learned but not yet
dominant" from "not learned at all" and exposes fixed-position shortcuts.
Inference only; no thresholds or gates live here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def choices_for(task):
    """Candidate strings for a task: listed entities if the answer is one, else listed values."""
    entities = [str(e) for e in task.get('entities', [])]
    values = [str(v) for v in task.get('values', [])]
    if task['answer'] in entities:
        return entities
    if task['answer'] in values:
        return values
    raise ValueError(f"Answer for {task['id']} is not among its listed entities or values")


def continuation_ids(tokenizer, prompt, choice):
    """Token ids of ' choice' after prompt, requiring the prompt tokenization to be a stable prefix."""
    prompt_ids = tokenizer.encode(prompt)
    full = tokenizer.encode(prompt + ' ' + choice)
    if full[:len(prompt_ids)] != prompt_ids or len(full) <= len(prompt_ids):
        raise ValueError(f'Unstable tokenization boundary for choice {choice!r}')
    return prompt_ids, full[len(prompt_ids):]


def summarize(records):
    """Aggregate per-category accuracy, both-variant pairs, margins and chosen-position counts."""
    groups = defaultdict(lambda: {'correct': 0, 'total': 0, 'chance': 0., 'margin_sum': 0.,
                                  'chosen_position': defaultdict(int), 'answer_position': defaultdict(int)})
    pairs = defaultdict(list)
    for row in records:
        g = groups[row['category']]
        g['correct'] += int(row['choice_correct'])
        g['total'] += 1
        g['chance'] += 1 / len(row['choice_logprobs'])
        g['margin_sum'] += row['answer_margin']
        g['chosen_position'][str(row['chosen_index'])] += 1
        g['answer_position'][str(row['answer_index'])] += 1
        if 'pair_id' in row:
            pairs[(row['category'], row['pair_id'])].append(row['choice_correct'])
    out = {}
    for key, g in groups.items():
        complete = [v for (category, _), v in pairs.items() if category == key and len(v) == 2]
        out[key] = {'correct': g['correct'], 'total': g['total'], 'accuracy': g['correct'] / g['total'],
                    'chance_accuracy': g['chance'] / g['total'], 'mean_answer_margin': g['margin_sum'] / g['total'],
                    'chosen_position': dict(g['chosen_position']), 'answer_position': dict(g['answer_position'])}
        if complete:
            out[key].update(complete_pairs=sum(all(v) for v in complete), pairs=len(complete))
    return out


def score_task(model, tokenizer, task):
    import mlx.core as mx
    import numpy as np
    choices = choices_for(task)
    logprobs = []
    for choice in choices:
        prompt_ids, target = continuation_ids(tokenizer, task['prompt'], choice)
        ids = prompt_ids + target
        logits, _, _ = model(mx.array([ids[:-1]]))
        rows = np.array(logits[0, len(prompt_ids) - 1:].astype(mx.float32))
        rows = rows - rows.max(axis=-1, keepdims=True)
        log_norm = np.log(np.exp(rows).sum(axis=-1))
        logprobs.append(float(sum(rows[i, t] - log_norm[i] for i, t in enumerate(target))))
    answer = choices.index(task['answer'])
    chosen = int(max(range(len(choices)), key=logprobs.__getitem__))
    rival = max(lp for i, lp in enumerate(logprobs) if i != answer)
    return {**task, 'choices': choices, 'choice_logprobs': logprobs, 'answer_index': answer,
            'chosen_index': chosen, 'choice_correct': chosen == answer, 'answer_margin': logprobs[answer] - rival}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; preserve earlier evidence and choose a new path')
    stopped = False

    def interrupt(_sig, _frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, interrupt)
    import mlx.core as mx
    from runtime.checkpoint_io import load_mlx_checkpoint_meta, validate_checkpoint_tokenizer
    from scripts.eval_fact_confidence import load_model
    from tokenizer.train_tokenizer import SpakieTokenizer
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    validate_checkpoint_tokenizer(load_mlx_checkpoint_meta(str(args.checkpoint)), str(args.tokenizer),
                                  source=str(args.checkpoint))
    model = load_model(args.checkpoint)
    mx.eval(model.parameters())
    tokenizer = SpakieTokenizer(str(args.tokenizer))
    tasks = json.loads(args.tasks.read_text())
    result = {'status': 'running', 'checkpoint_sha256': digest(args.checkpoint), 'tokenizer_sha256': digest(args.tokenizer),
              'tasks_sha256': digest(args.tasks), 'script_sha256': digest(__file__), 'records': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        for task in tasks:
            if stopped:
                break
            result['records'].append(score_task(model, tokenizer, task))
        result['status'] = 'interrupted' if stopped else 'complete'
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['summary'] = summarize(result['records'])
        result['elapsed_seconds'] = time.monotonic() - start
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: (v['correct'], v['total'], v.get('complete_pairs')) for k, v in result['summary'].items()}), flush=True)
    return 130 if stopped else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
