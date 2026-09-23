"""Canonical-answer confidence for the frozen BASE fact probes.

Supplements the brittle greedy fact checks in scripts/_base_probe_cases.py with
teacher-forced conditional log probabilities of ONE maintained canonical answer
per fact, plus margins against rival tokens frozen from the parent checkpoint.

Diagnostic only: this is not an acceptance gate and not a semantic-correctness
claim. Valid paraphrases ("100 °C", "the Earth") are not scored, and a negative
margin only means the parent's rival token outscored the canonical token at
that position. These canonical facts must never be added to training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
SCOPE = ('Diagnostic canonical-answer likelihood for the 24 frozen BASE fact probes. '
         'Scores only the maintained canonical reference text; valid paraphrases are not '
         'scored. Rival tokens are the parent checkpoint\'s highest-logit non-reference '
         'tokens under the canonical prefix, frozen for every candidate. Negative margins '
         'are not evidence of a semantically incorrect answer. Not an acceptance gate.')

# Maintained canonical reference answer per fact ID. Each must satisfy that
# fact's expected_regex; tests enforce coverage against suite().
CANONICAL_ANSWERS = {
    'fact_france': 'Paris',
    'fact_japan': 'Tokyo',
    'fact_italy': 'Rome',
    'fact_germany': 'Berlin',
    'fact_uk': 'London',
    'fact_australia': 'Canberra',
    'fact_water': 'H2O',
    'fact_gold': 'Au',
    'fact_oxygen': 'O',
    'fact_earth': 'Earth',
    'fact_mars': 'Mars',
    'fact_largest_planet': 'Jupiter',
    'fact_moon': 'Moon',
    'fact_sun': 'Sun',
    'fact_heart': 'heart',
    'fact_lungs': 'lungs',
    'fact_photosynthesis': 'photosynthesis',
    'fact_freeze': '0 degrees Celsius',
    'fact_boil': '100 degrees Celsius',
    'fact_week': 'week',
    'fact_year': 'year',
    'fact_triangle': 'three sides',
    'fact_hamlet': 'William Shakespeare',
    'fact_pride': 'Jane Austen',
}

# Config fields that determine model semantics; training knobs may differ.
MODEL_CONFIG_KEYS = (
    'vocab_size', 'n_layers', 'n_heads', 'n_kv_heads', 'd_model', 'd_ff', 'mlp_type',
    'norm_type', 'qk_norm', 'position_encoding', 'rope_theta', 'residual_type',
    'swiglu_hidden', 'max_seq_len', 'bias',
)


def fact_cases():
    from scripts._base_probe_cases import suite
    facts = [r for r in suite() if r['category'] == 'facts']
    ids = [r['id'] for r in facts]
    if sorted(ids) != sorted(CANONICAL_ANSWERS):
        raise ValueError('CANONICAL_ANSWERS does not match the fact IDs in _base_probe_cases.suite()')
    return [dict(id=r['id'], prompt=r['prompt'], expected_regex=r['expected_regex'],
                 canonical_answer=CANONICAL_ANSWERS[r['id']]) for r in facts]


# ── Pure token-level helpers ──────────────────────────────────────────────

def answer_ids_from_alignment(prompt_ids, combined_ids, *, fact_id=''):
    """Return answer tokens, requiring the prompt tokens to prefix the combined tokens."""
    prompt_ids, combined_ids = list(prompt_ids), list(combined_ids)
    p = len(prompt_ids)
    if p == 0:
        raise ValueError(f'{fact_id}: prompt tokenizes to no tokens')
    if combined_ids[:p] != prompt_ids:
        raise ValueError(f'{fact_id}: prompt tokens {prompt_ids} are not an exact prefix of '
                         f'prompt+answer tokens {combined_ids}; refusing to score different text')
    if len(combined_ids) == p:
        raise ValueError(f'{fact_id}: canonical answer adds no tokens')
    return combined_ids[p:]


def answer_logit_rows(logits, prompt_len, n_answer):
    """Rows of `logits` (over input combined[:-1]) that predict each answer token."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != prompt_len + n_answer - 1:
        raise ValueError(f'Expected {prompt_len + n_answer - 1} logit rows, got shape {logits.shape}')
    return logits[prompt_len - 1:]


def log_softmax(rows):
    rows = np.asarray(rows, dtype=np.float64)
    shifted = rows - rows.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def select_rivals(rows, ref_ids):
    """Highest-logit token at each position, excluding that position's reference token."""
    masked = np.array(rows, dtype=np.float64)
    positions = np.arange(len(ref_ids))
    masked[positions, np.asarray(ref_ids)] = -np.inf
    return [int(i) for i in masked.argmax(axis=-1)]


def token_stats(rows, ref_ids, rival_ids):
    rows = np.asarray(rows, dtype=np.float64)
    if not len(ref_ids) == len(rival_ids) == rows.shape[0]:
        raise ValueError('Answer, rival, and logit-row lengths differ')
    if any(r == v for r, v in zip(ref_ids, rival_ids)):
        raise ValueError('A rival token equals its reference token')
    lp = log_softmax(rows)
    pos = np.arange(len(ref_ids))
    ref_lp, rival_lp = lp[pos, ref_ids], lp[pos, rival_ids]
    margin = ref_lp - rival_lp
    positions = [dict(ref_logprob=float(a), rival_logprob=float(b), margin=float(m),
                      ref_is_argmax=bool(rows[i, ref_ids[i]] >= rows[i].max()))
                 for i, (a, b, m) in enumerate(zip(ref_lp, rival_lp, margin))]
    return dict(positions=positions, sum_ref_logprob=float(ref_lp.sum()),
                mean_ref_logprob=float(ref_lp.mean()), mean_margin=float(margin.mean()),
                min_margin=float(margin.min()),
                all_ref_argmax=all(p['ref_is_argmax'] for p in positions))


SUMMARY_KEYS = ('sum_ref_logprob', 'mean_ref_logprob', 'mean_margin', 'min_margin')


def stat_changes(parent, candidate):
    return dict({f'delta_{k}': candidate[k] - parent[k] for k in SUMMARY_KEYS},
                delta_positions=[dict(ref_logprob=c['ref_logprob'] - p['ref_logprob'],
                                      margin=c['margin'] - p['margin'])
                                 for p, c in zip(parent['positions'], candidate['positions'])])


def aggregate(items):
    if not items:
        return {}
    stats = [i['stats'] for i in items]
    out = {f'mean_{k}': float(np.mean([s[k] for s in stats])) for k in SUMMARY_KEYS}
    out.update(n_facts=len(stats),
               n_min_margin_positive=sum(s['min_margin'] > 0 for s in stats),
               n_all_ref_argmax=sum(s['all_ref_argmax'] for s in stats))
    if all('changes' in i for i in items):
        out.update({f'mean_delta_{k}': float(np.mean([i['changes'][f'delta_{k}'] for i in items]))
                    for k in SUMMARY_KEYS})
    return out


def config_mismatches(parent_cfg, candidate_cfg):
    return [k for k in MODEL_CONFIG_KEYS if parent_cfg.get(k) != candidate_cfg.get(k)]


def check_metadata(parent_meta, candidate_meta, *, source, parent_source=None):
    """Raise unless both are pretrain checkpoints with matching model/tokenizer config."""
    from scripts._base_probe_cases import is_base_checkpoint
    for label, meta, path in (('parent', parent_meta, parent_source), (source, candidate_meta, source)):
        if not is_base_checkpoint(meta, path):
            raise ValueError(f'{label}: checkpoint metadata must identify stage "pretrain"')
        if not isinstance(meta.get('config'), dict):
            raise ValueError(f'{label}: checkpoint has no full configuration metadata')
    bad = config_mismatches(parent_meta['config'], candidate_meta['config'])
    if bad:
        raise ValueError(f'{source}: model configuration differs from parent in {bad}')
    if parent_meta.get('tokenizer') != candidate_meta.get('tokenizer'):
        raise ValueError(f'{source}: tokenizer contract differs from parent')


# ── Output ────────────────────────────────────────────────────────────────

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def create_output(path):
    """Claim a new output path; never overwrite existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x'):
        pass


def write_json(path, value):
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


# ── MLX runtime ───────────────────────────────────────────────────────────

def load_model(path):
    from runtime.checkpoint_io import load_mlx_checkpoint_config, load_mlx_model_weights_strict
    from runtime.mlx_backend import load_safetensors
    from model.transformer_mlx import SpakieGPTMLX
    model = SpakieGPTMLX(load_mlx_checkpoint_config(str(path)))
    load_mlx_model_weights_strict(model, load_safetensors(str(path)), path=str(path))
    model.eval()
    return model


def answer_rows(model, prompt_len, combined_ids):
    import mlx.core as mx
    logits, _, _ = model(mx.array([combined_ids[:-1]]))
    return answer_logit_rows(np.array(logits[0].astype(mx.float32)), prompt_len,
                             len(combined_ids) - prompt_len)


def score_model(model, cases, items, *, rivals=None, parent_items=None, on_item, stopped):
    """Fill `items` in place; derive rivals from this model when `rivals` is None."""
    for i, case in enumerate(cases):
        if stopped():
            return False
        rows = answer_rows(model, len(case['prompt_ids']), case['combined_ids'])
        rival_ids = select_rivals(rows, case['answer_ids']) if rivals is None else rivals[i]
        item = dict(id=case['id'], stats=token_stats(rows, case['answer_ids'], rival_ids))
        if rivals is None:
            item['rival_ids'] = rival_ids
        if parent_items is not None:
            item['changes'] = stat_changes(parent_items[i]['stats'], item['stats'])
        items.append(item)
        on_item()
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True, help='BASE pretrain checkpoint (.safetensors)')
    p.add_argument('--checkpoint', type=Path, action='append', required=True,
                   help='Candidate pretrain checkpoint; repeatable')
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Output already exists')
    paths = [args.parent.resolve()] + [c.resolve() for c in args.checkpoint]
    if len(set(paths)) != len(paths):
        p.error('Parent and candidate checkpoints must be distinct')

    from runtime.checkpoint_io import load_mlx_checkpoint_meta, validate_checkpoint_tokenizer
    from tokenizer.train_tokenizer import SpakieTokenizer
    metas = [load_mlx_checkpoint_meta(str(path)) for path in paths]
    for path, meta in zip(paths, metas):
        check_metadata(metas[0], meta, source=str(path), parent_source=str(paths[0]))
        validate_checkpoint_tokenizer(meta, str(args.tokenizer), source=str(path))
    if metas[0]['config'].get('vocab_size') != metas[0]['tokenizer'].get('vocab_size'):
        p.error('Checkpoint vocab_size does not match its tokenizer contract')

    tok = SpakieTokenizer(str(args.tokenizer))
    cases = []
    for case in fact_cases():
        prompt_ids = tok.encode(case['prompt'])
        combined_ids = tok.encode(case['prompt'] + ' ' + case['canonical_answer'])
        answer_ids = answer_ids_from_alignment(prompt_ids, combined_ids, fact_id=case['id'])
        cases.append(dict(case, prompt_ids=prompt_ids, combined_ids=combined_ids, answer_ids=answer_ids,
                          answer_pieces=[tok.id_to_piece(t) for t in answer_ids]))

    create_output(args.output)
    result = dict(
        schema_version=SCHEMA_VERSION, scope=SCOPE, status='running',
        script_sha256=sha(__file__), cases_sha256=hashlib.sha256(
            json.dumps(fact_cases(), sort_keys=True).encode()).hexdigest(),
        tokenizer=dict(path=str(args.tokenizer), sha256=sha(args.tokenizer), contract=metas[0]['tokenizer']),
        cases=[{k: c[k] for k in ('id', 'prompt', 'canonical_answer', 'prompt_ids', 'answer_ids',
                                  'answer_pieces')} for c in cases],
        parent=dict(path=str(paths[0]), sha256=sha(paths[0]), items=[], aggregate={}),
        candidates=[])
    save = lambda: write_json(args.output, result)
    save()

    stop = False
    def interrupt(_sig, _frame):
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        parent = result['parent']
        model = load_model(paths[0])
        done = score_model(model, cases, parent['items'], on_item=save, stopped=lambda: stop)
        del model
        parent['aggregate'] = aggregate(parent['items'])
        for item in parent['items']:
            item['rival_pieces'] = [tok.id_to_piece(t) for t in item['rival_ids']]
        rivals = [item['rival_ids'] for item in parent['items']]
        for path in paths[1:] if done else []:
            if stop:
                break
            entry = dict(path=str(path), sha256=sha(path), items=[], aggregate={})
            result['candidates'].append(entry)
            model = load_model(path)
            done = score_model(model, cases, entry['items'], rivals=rivals, parent_items=parent['items'],
                               on_item=save, stopped=lambda: stop)
            del model
            entry['aggregate'] = aggregate(entry['items'])
            if not done:
                break
        complete = not stop and len(result['candidates']) == len(paths) - 1 and done
    except KeyboardInterrupt:  # second Ctrl+C: abandon the in-flight item
        complete = False
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
    try:
        result['status'] = 'complete' if complete else 'interrupted'
        save()
        print(json.dumps(dict(status=result['status'], parent=parent['aggregate'],
                              candidates=[dict(path=c['path'], aggregate=c['aggregate'])
                                          for c in result['candidates']])), flush=True)
        return 0 if complete else 130
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
