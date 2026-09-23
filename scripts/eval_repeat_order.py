"""Bounded MLX diagnostic for predicting repeated versus shuffled token sequences.

This artificial copying task does not qualify reasoning, retrieval, or a model
recipe. Cross-model comparisons still depend on training history and vocabulary.
Prepare one frozen protocol from every compared tokenizer, then reuse it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA_VERSION = 1


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build_protocol(piece_sets, *, seed=2026092318, pool_size=256,
                   lengths=(16, 32, 64), cases_per_length=32):
    if not piece_sets:
        raise ValueError('Provide at least one tokenizer')
    if not lengths or len(set(lengths)) != len(lengths) or any(n < 3 or n > 128 for n in lengths):
        raise ValueError('Use distinct sequence lengths between 3 and 128')
    if not 1 <= cases_per_length <= 256:
        raise ValueError('Cases per length must be between 1 and 256')
    common = sorted(set.intersection(*[set(pieces) for pieces in piece_sets]))
    common = [piece for piece in common if re.fullmatch('▁[a-z]{2,10}', piece)]
    if not max(lengths) <= pool_size <= len(common):
        raise ValueError('Pool must cover the longest sequence and fit the common vocabulary')
    rng = np.random.default_rng(seed)
    pool = sorted(str(x) for x in rng.choice(common, pool_size, replace=False))
    cases = []
    for length in lengths:
        for index in range(cases_per_length):
            first = rng.choice(pool_size, length, replace=False).tolist()
            for _ in range(64):
                order = [0] + rng.permutation(np.arange(1, length)).tolist()
                if all(b != a + 1 for a, b in zip(order, order[1:])):
                    break
            else:
                order = [0] + list(range(length - 1, 0, -1))
            cases.append(dict(id=f'L{length}_{index}', length=length, first=first,
                              repeat=first, shuffled=[first[i] for i in order]))
    return dict(schema_version=SCHEMA_VERSION, seed=seed, pool_pieces=pool,
                common_piece_count=len(common), cases=cases,
                scope='Artificial repeat-order diagnostic only; not a model-quality acceptance gate.')


def validate_protocol(protocol):
    if protocol.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported repeat-order protocol schema')
    pool = protocol['pool_pieces']
    if not pool or len(pool) != len(set(pool)) or any(not isinstance(p, str) for p in pool):
        raise ValueError('Protocol token pieces must be unique strings')
    cases = protocol['cases']
    if not cases or len(cases) > 32768 or len({row['id'] for row in cases}) != len(cases):
        raise ValueError('Protocol must contain bounded, uniquely identified cases')
    for row in cases:
        n = row['length']; first = row['first']; shuffled = row['shuffled']
        if not isinstance(n, int) or not 3 <= n <= 128:
            raise ValueError('Protocol sequence lengths must be between 3 and 128')
        if (len(first) != n or len(set(first)) != n
                or any(not isinstance(i, int) or not 0 <= i < len(pool) for i in first)
                or row['repeat'] != first or sorted(first) != sorted(shuffled)
                or first[0] != shuffled[0]):
            raise ValueError('Protocol must preserve the unique token multiset and first token')
        if set(zip(first, first[1:])) & set(zip(shuffled, shuffled[1:])):
            raise ValueError('Shuffled control must exclude original adjacent transitions')


def summarize_prediction_rows(nll, correct, length):
    """Rows predict targets of EOS+A+B; omit the first target of each sequence."""
    nll, correct = np.asarray(nll), np.asarray(correct)
    if nll.shape != (2, 2 * length) or correct.shape != nll.shape:
        raise ValueError('Expected repeat/control rows for both complete sequences')
    if not np.all(np.isfinite(nll)):
        raise ValueError('Nonfinite prediction loss')
    first, second = slice(1, length), slice(length + 1, 2 * length)
    return dict(scored_tokens=length - 1,
                first_nll=float(nll[0, first].mean()),
                repeat_nll=float(nll[0, second].mean()),
                shuffle_nll=float(nll[1, second].mean()),
                first_correct=int(correct[0, first].sum()),
                repeat_correct=int(correct[0, second].sum()),
                shuffle_correct=int(correct[1, second].sum()),
                first_condition_max_nll_delta=float(np.max(np.abs(nll[0, first] - nll[1, first]))))


def aggregate_records(records):
    metrics = {}
    for length in sorted({row['length'] for row in records}):
        rows = [row for row in records if row['length'] == length]
        tokens = sum(row['scored_tokens'] for row in rows)
        means = {key: float(np.mean([row[key] for row in rows]))
                 for key in ('first_nll', 'repeat_nll', 'shuffle_nll')}
        metrics[str(length)] = dict(cases=len(rows), scored_tokens=tokens, **means,
            shuffle_minus_repeat_nll=means['shuffle_nll'] - means['repeat_nll'],
            **{key.removesuffix('_correct') + '_accuracy': sum(row[key] for row in rows) / tokens
               for key in ('first_correct', 'repeat_correct', 'shuffle_correct')})
    return metrics


def run(args):
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    start = time.monotonic()
    result = dict(status='loading', protocol_sha256=digest(args.protocol),
                  checkpoint_sha256=digest(args.checkpoint),
                  tokenizer_sha256=digest(args.tokenizer), script_sha256=digest(__file__),
                  scope=protocol.get('scope'), records=[])
    try:
        import mlx.core as mx
        import sentencepiece as spm
        from runtime.checkpoint_io import (load_mlx_checkpoint_config, load_mlx_checkpoint_meta,
            load_mlx_model_weights_strict, validate_checkpoint_tokenizer)
        from runtime.mlx_backend import load_safetensors
        from model.transformer_mlx import SpakieGPTMLX

        meta = load_mlx_checkpoint_meta(str(args.checkpoint))
        validate_checkpoint_tokenizer(meta, str(args.tokenizer), source=str(args.checkpoint))
        config = load_mlx_checkpoint_config(str(args.checkpoint))
        if max(2 * row['length'] for row in protocol['cases']) > config.max_seq_len:
            raise ValueError('Protocol exceeds this checkpoint context limit; prepare shorter sequences')
        tok = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        pool = [tok.piece_to_id(piece) for piece in protocol['pool_pieces']]
        if any(tok.id_to_piece(i) != piece for i, piece in zip(pool, protocol['pool_pieces'])):
            raise ValueError('Protocol includes a piece absent from this tokenizer')
        model = SpakieGPTMLX(config)
        load_mlx_model_weights_strict(model, load_safetensors(str(args.checkpoint)), path=str(args.checkpoint))
        model.eval()
        mx.eval(model.parameters())
        result.update(status='running', context_limit=config.max_seq_len, vocab_size=config.vocab_size)
        for case in protocol['cases']:
            if stopped or time.monotonic() - start >= args.seconds:
                break
            full = [[tok.eos_id()] + [pool[i] for i in case['first'] + case[k]]
                    for k in ('repeat', 'shuffled')]
            arrays = np.asarray(full, dtype=np.int32)
            logits, _, _ = model(mx.array(arrays[:, :-1]))
            logits = logits.astype(mx.float32)
            targets = mx.array(arrays[:, 1:])
            nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(
                logits, targets[..., None], axis=-1).squeeze(-1)
            correct = mx.argmax(logits, axis=-1) == targets
            mx.eval(nll, correct)
            result['records'].append(dict(id=case['id'], length=case['length'],
                **summarize_prediction_rows(np.asarray(nll), np.asarray(correct), case['length'])))
            del logits, targets, nll, correct
        complete = len(result['records']) == len(protocol['cases'])
        result['status'] = 'interrupted' if stopped else 'complete' if complete else 'time_limit'
        result['peak_gib'] = mx.get_peak_memory() / 1024**3
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        result['metrics'] = aggregate_records(result['records'])
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds')}), flush=True)
    return 130 if stopped else 0 if result['status'] == 'complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--tokenizers', type=Path, nargs='+', required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--seed', type=int, default=2026092318)
    prepare.add_argument('--pool-size', type=int, default=256)
    prepare.add_argument('--lengths', type=int, nargs='+', default=[16, 32, 64])
    prepare.add_argument('--cases-per-length', type=int, default=32)
    evaluate = sub.add_parser('run')
    for name in ('protocol', 'checkpoint', 'tokenizer', 'output'):
        evaluate.add_argument('--' + name, type=Path, required=True)
    evaluate.add_argument('--seconds', type=float, default=120.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path to preserve earlier evidence')
    if args.command == 'run' and (not np.isfinite(args.seconds) or not 0 < args.seconds <= 600):
        parser.error('--seconds must be finite and in (0, 600]')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'run':
        return run(args)
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be in [0, 2**32)')
    import sentencepiece as spm
    tokenizers = [spm.SentencePieceProcessor(model_file=str(path)) for path in args.tokenizers]
    try:
        protocol = build_protocol([{tok.id_to_piece(i) for i in range(tok.vocab_size())}
            for tok in tokenizers], seed=args.seed, pool_size=args.pool_size,
            lengths=args.lengths, cases_per_length=args.cases_per_length)
    except ValueError as exc:
        parser.error(str(exc))
    protocol['tokenizers'] = [{'path':str(path.resolve()), 'sha256':digest(path)} for path in args.tokenizers]
    validate_protocol(protocol)
    args.output.write_text(json.dumps(protocol, indent=2) + '\n')
    print(f"Prepared {len(protocol['cases'])} paired cases", flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nStopped.', file=sys.stderr)
        raise SystemExit(130)
