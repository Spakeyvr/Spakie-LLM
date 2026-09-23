"""Bounded from-scratch pretraining comparisons; generated outputs stay local.

Prepare a small, document-disjoint source sample, then compare recipes with
identical initialization and sampling seeds. These are learning diagnostics,
not evidence that a tiny pilot has acquired general language capabilities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ('fineweb-edu', 'wikipedia_snapshot', 'finemath', 'python_edu', 'cosmopedia_v2')
MIXES = {'baseline': (.45, .15, .15, .15, .10),
         'education': (.30, .15, .10, .10, .35)}
MAX_STEPS = 512
MAX_SECONDS = 180


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def text_key(text):
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()


def validate_args(args):
    seconds_limit = 900 if args.long_context_pilot else MAX_SECONDS
    if not 1 <= args.steps <= MAX_STEPS or not 1 <= args.seconds <= seconds_limit:
        raise ValueError(f'Runs are limited to {MAX_STEPS} updates / {seconds_limit} training seconds')
    if not math.isfinite(args.lr) or not 0 < args.lr <= .01:
        raise ValueError('Learning rate must be finite and in (0, .01]')
    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be in [0, 2**32)')
    if args.output.resolve().is_relative_to(args.assets.resolve()):
        raise ValueError('Output must be outside the original asset tree')
    if args.output.exists():
        raise ValueError('Output already exists; choose a new directory')
    if args.action == 'run' and args.data is None:
        raise ValueError('--data is required for run')
    if args.scale == '360m' and args.steps > 64 and not args.long_context_pilot:
        raise ValueError('Target-size probes are limited to 64 updates')
    if args.scale == '360m' and args.batch_size > 2:
        raise ValueError('Target-size probes are limited to batches of two')


def prepare(args):
    import sentencepiece as spm
    from configs.default import get_preset_config
    from runtime.langid import is_probably_english, load_langid_model
    from scripts.prepare_data import (
        PREPARATION_SCHEMA_VERSION, clean_text, should_keep_document, language_filter_sample,
        NearDuplicateIndex, compute_exact_hash, compute_minhash_signature,
    )
    cfg = get_preset_config('92m')
    sources = tuple(s for s,p in cfg.corpus_source_plan.items()
                    if p.get('enabled') and p['target_tokens'] > 0) if args.production_mix else SOURCES
    cfg.langid_model_path = str(args.assets / 'data/models/lid.176.ftz')
    if not Path(cfg.langid_model_path).is_file():
        raise ValueError('A local language-ID model is required; this probe never downloads data')
    if load_langid_model(cfg) is None:
        raise ValueError('Language-ID model unavailable')
    args.output.mkdir(parents=True)
    original = args.assets / 'tokenizer/spakie.model'
    tok = spm.SentencePieceProcessor(model_file=str(original))
    pieces = [(tok.id_to_piece(i), tok.get_score(i)) for i in range(tok.vocab_size())]
    tok.override_normalizer_spec(remove_extra_whitespaces=False)
    model_path = args.output / 'tokenizer.model'
    model_path.write_bytes(tok.serialized_model_proto())
    tok = spm.SentencePieceProcessor(model_file=str(model_path))
    assert pieces == [(tok.id_to_piece(i), tok.get_score(i)) for i in range(tok.vocab_size())]
    manifest = {'seed': 922, 'sources': {}, 'files': {}, 'vocab_size': tok.vocab_size(),
                'preparation_schema_version': PREPARATION_SCHEMA_VERSION,
                'preparation_script_sha256': sha(ROOT / 'scripts/prepare_data.py'),
                'original_tokenizer_sha256': sha(original), 'tokenizer_sha256': sha(model_path),
                'source_order': sources,
                'production_weights': [cfg.corpus_source_plan[s]['target_tokens'] for s in sources],
                'scope': 'Local source sample with disjoint shards and normalized exact dedup. '
                         'Production mode also uses canonical MinHash/LSH within this sample. '
                         'Original raw code may already have lost whitespace.'}
    seen = set()
    near = NearDuplicateIndex(threshold=cfg.near_dup_jaccard_threshold,
        num_perm=cfg.near_dup_num_perm, shingle_size=cfg.near_dup_shingle_size) if args.production_mix else None
    rng = np.random.default_rng(922)
    try:
        with (args.output / 'documents.jsonl').open('w') as docs:
            for source in sources:
                paths = sorted((args.assets / 'data/raw/large_corpus' / source).glob('*.jsonl'))
                if len(paths) < 6:
                    raise ValueError(f'Insufficient shards for {source}')
                indices = rng.choice(len(paths), size=min(16, len(paths)), replace=False)
                stats = Counter()
                buffers = {'train': [], 'dev': [], 'test': []}
                # Separate raw shards before filtering or tokenization.
                for j, index in enumerate(indices):
                    split = 'dev' if j < 2 else ('test' if j < 4 else 'train')
                    cap = 512_000 if split == 'train' else 32_768
                    shard_cap = cap // (len(indices) - 4 if split == 'train' else 2)
                    shard_start = len(buffers[split])
                    path = paths[int(index)]
                    with path.open() as stream:
                        for line_number, line in enumerate(stream, 1):
                            if line_number > 512 or len(buffers[split]) - shard_start >= shard_cap:
                                break
                            row = json.loads(line)
                            raw = row.get('text', '')
                            stats['seen'] += 1
                            if not isinstance(raw, str) or len(raw) > 100_000:
                                stats['oversized_or_invalid'] += 1
                                continue
                            text = clean_text(raw, source)
                            keep, reason = should_keep_document(text, cfg, source)
                            language = language_filter_sample(text, cfg, source)
                            if keep and language is not None:
                                keep = is_probably_english(language, cfg)
                                if not keep: reason = 'language'
                            if not keep:
                                stats['rejected_' + reason] += 1
                                continue
                            key = text_key(text)
                            if key in seen:
                                stats['duplicate'] += 1
                                continue
                            if near is not None and near.query_signature(compute_exact_hash(text),
                                    compute_minhash_signature(text, num_perm=cfg.near_dup_num_perm,
                                                              shingle_size=cfg.near_dup_shingle_size)):
                                stats['near_duplicate'] += 1
                                continue
                            seen.add(key)
                            # Bound document influence, especially on validation.
                            encoded = tok.encode(text)
                            limit = (8192 if split == 'train' else 4096) if args.production_mix else (4096 if split == 'train' else 1024)
                            ids = encoded[:limit] + [tok.eos_id()]
                            stats['truncated_docs'] += int(len(encoded) > limit)
                            buffers[split].extend(ids)
                            stats[split + '_docs'] += 1
                            docs.write(json.dumps({'source': source, 'split': split, 'path': str(path),
                                                   'line': line_number, 'sha256': key, 'tokens': len(ids)}) + '\n')
                    stats[split + '_shards'] += 1
                for split, ids in buffers.items():
                    if len(ids) < 8193:
                        raise ValueError(f'Too few tokens for {source}/{split}: {len(ids)}')
                    path = args.output / f'{source}_{split}.npy'
                    np.save(path, np.asarray(ids, dtype=np.uint16))
                    manifest['files'][path.name] = sha(path)
                    stats[split + '_tokens'] = len(ids)
                manifest['sources'][source] = dict(stats)
                print(source, dict(stats), flush=True)
        manifest['documents_sha256'] = sha(args.output / 'documents.jsonl')
        manifest['status'] = 'complete'
    finally:
        write_json(args.output / 'manifest.json', manifest)


def batch_from_streams(streams, mix, seed, steps, batch_size=8, seq_len=256, weights=None,
                       *, independent_source_rng=False):
    rng = np.random.default_rng(seed)
    if independent_source_rng:
        source_seed, offset_seed = np.random.SeedSequence(seed).spawn(2)
        source_rng = np.random.default_rng(source_seed)
        rng = np.random.default_rng(offset_seed)
    else:
        source_rng = rng
    sources = tuple(streams)
    weights = np.asarray(weights if weights is not None else MIXES[mix], dtype=float)
    weights /= weights.sum()
    for _ in range(steps):
        draws = source_rng.choice(len(sources), size=batch_size, p=weights)
        batch = np.empty((batch_size, seq_len + 1), dtype=np.int32)
        for row, index in enumerate(draws):
            stream = streams[sources[int(index)]]
            start = int(rng.integers(0, len(stream) - seq_len))
            batch[row] = stream[start:start + seq_len + 1]
        yield batch, draws


def run(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from configs.default import config_to_dict, get_preset_config
    from model.transformer_mlx import SpakieGPTMLX
    from runtime.mlx_backend import clip_grads
    from training.optimizers_mlx import configure_mlx_optimizer
    from training.pretrain_mlx import _build_microbatch_step, get_lr
    manifest = json.loads((args.data / 'manifest.json').read_text())
    sources = tuple(manifest.get('source_order', SOURCES))
    weights = manifest['production_weights'] if args.production_mix else MIXES[args.mix]
    if len(weights) != len(sources):
        raise ValueError('Source count requires --production-mix')
    if manifest.get('status') != 'complete':
        raise ValueError('Pilot dataset preparation did not complete')
    for name, expected in manifest['files'].items():
        if sha(args.data / name) != expected:
            raise ValueError(f'Pilot dataset changed: {name}')
    if sha(args.data / 'tokenizer.model') != manifest['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    args.output.mkdir(parents=True)
    cfg = get_preset_config('300m' if args.scale == '360m' else '92m')
    if args.scale == 'small':
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads = 6, 384, 6, 2
        cfg.d_ff = cfg.swiglu_hidden = 1024
    elif args.scale == '360m':
        cfg.n_layers = 28
    cfg.vocab_size = manifest['vocab_size']
    cfg.max_seq_len = 2048 if args.long_context_pilot else 256
    cfg.pretrain_batch_size = args.batch_size or (8 if args.scale == 'small' else 2)
    cfg.pretrain_grad_accum_steps = 1
    cfg.pretrain_target_tokens = args.steps * cfg.pretrain_tokens_per_step()
    cfg.refresh_derived_fields()
    cfg.pretrain_warmup_steps = max(1, round(args.steps * .05))
    cfg.pretrain_lr = args.lr
    cfg.pretrain_lr_schedule = 'trapezoid'
    cfg.muon_ns_steps = args.ns_steps
    cfg.tokenizer_prefix = str(args.data.resolve() / 'tokenizer')
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(args.seed)
    model = SpakieGPTMLX(cfg)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    initial = hashlib.sha256()
    for key, value in tree_flatten(model.parameters()):
        initial.update(key.encode())
        initial.update(np.asarray(value.astype(mx.float32)).tobytes())
    optimizer = configure_mlx_optimizer(model, cfg, kind=args.optimizer,
                                         learning_rate=args.lr, weight_decay=.1)
    step_fn = _build_microbatch_step(model, 1., compile_step=False, ignore_index=None)
    streams = {source: np.load(args.data / f'{source}_train.npy', mmap_mode='r') for source in sources}
    summary = {'status': 'running', 'completed_steps': 0, 'seed': args.seed, 'mix': args.mix,
               'independent_source_rng': args.independent_source_rng,
               'optimizer': args.optimizer, 'lr': args.lr, 'ns_steps': args.ns_steps,
               'scale': args.scale, 'parameters': sum(x.size for _, x in tree_flatten(model.parameters())),
               'initial_weights_sha256': initial.hexdigest(), 'manifest_sha256': sha(args.data / 'manifest.json'),
               'script_sha256': sha(__file__), 'max_training_seconds': args.seconds, 'max_steps': args.steps,
               'source_weights': list(weights), 'config': config_to_dict(cfg), 'evaluations': [],
               'sampled_source_tokens': {s: 0 for s in sources}}
    (args.output / 'script.py').write_text(Path(__file__).read_text())
    stop = False
    def interrupt(_sig, _frame):
        nonlocal stop
        stop = True
    old = signal.signal(signal.SIGINT, interrupt)
    started = time.monotonic()
    training_seconds = 0.
    def evaluate(split, step):
        model.eval()
        losses = {}
        for source in sources:
            data = np.load(args.data / f'{source}_{split}.npy', mmap_mode='r')
            rng = np.random.default_rng(713 if split == 'dev' else 991)
            length = cfg.max_seq_len
            count = 4 if args.long_context_pilot else 16
            starts = rng.choice((len(data) - 1) // length, size=count, replace=False) * length
            batch = np.array([data[i:i+length+1] for i in starts], dtype=np.int32)
            total = []
            for i in range(0, len(batch), 2):
                if stop: return
                _, loss, _ = model(mx.array(batch[i:i+2, :-1]), mx.array(batch[i:i+2, 1:]), ignore_index=None)
                total.append(float(loss.item()))
            losses[source] = float(np.mean(total))
        result = {'split': split, 'step': step, 'losses': losses,
                  'macro_nll': float(np.mean(list(losses.values())))}
        summary['evaluations'].append(result)
        print(json.dumps(result), flush=True)
        model.train()
    try:
        with (args.output / 'training.jsonl').open('w') as journal:
            evaluate('dev', 0)
            for step, (batch, draws) in enumerate(batch_from_streams(streams, args.mix, args.seed, args.steps,
                                                                     cfg.pretrain_batch_size, cfg.max_seq_len, weights,
                                                                     independent_source_rng=args.independent_source_rng), 1):
                if stop or training_seconds >= args.seconds:
                    break
                tick = time.monotonic()
                optimizer.set_lr(get_lr(step - 1, cfg))
                loss, grads = step_fn(mx.array(batch[:, :-1]), mx.array(batch[:, 1:]))
                grads, norm = clip_grads(grads, 1.)
                mx.eval(loss, norm)
                lv, nv = float(loss.item()), float(norm.item())
                if not math.isfinite(lv) or not math.isfinite(nv):
                    raise ValueError('Non-finite loss or gradient')
                optimizer.update(model, grads)
                optimizer.eval_state()
                training_seconds += time.monotonic() - tick
                summary['completed_steps'] = step
                for index in draws:
                    summary['sampled_source_tokens'][sources[int(index)]] += cfg.max_seq_len
                journal.write(json.dumps({'step': step, 'loss': lv, 'gradient_norm': nv,
                    'lr': float(get_lr(step-1,cfg)), 'batch_sha256': hashlib.sha256(batch.tobytes()).hexdigest(),
                    'training_seconds': training_seconds}) + '\n')
                journal.flush()
                if step % 64 == 0:
                    print(f'step {step}/{args.steps} loss={lv:.4f} training={training_seconds:.1f}s', flush=True)
                if step % 128 == 0:
                    evaluate('dev', step)
            if not stop:
                evaluate('dev', summary['completed_steps'])
                if args.final_test:
                    evaluate('test', summary['completed_steps'])
                if args.capabilities:
                    from scripts.eval_base_readiness import evaluate_model
                    summary['capabilities'] = evaluate_model(model, args.data / 'tokenizer.model',
                        args.output / 'capabilities.json', split='dev', stopped=lambda: stop)
                if args.save_model and not stop:
                    from runtime.mlx_backend import save_safetensors_checkpoint
                    from runtime.checkpoint_io import checkpoint_tokenizer_contract
                    from configs.default import CHECKPOINT_CONFIG_SCHEMA_VERSION
                    save_safetensors_checkpoint(str(args.output / 'base.safetensors'),
                        {'model.' + k: v for k, v in tree_flatten(model.parameters())},
                        {'config': config_to_dict(cfg), 'config_schema_version': CHECKPOINT_CONFIG_SCHEMA_VERSION,
                         'tokenizer': checkpoint_tokenizer_contract(cfg), 'stage': 'pretrain',
                         'step': summary['completed_steps'], 'experiment': {'inference_only': True,
                         'data_manifest_sha256': summary['manifest_sha256'],
                         'script_sha256': summary['script_sha256']}})
                    summary['checkpoint_sha256'] = sha(args.output / 'base.safetensors')
            summary['status'] = 'interrupted' if stop else ('complete' if summary['completed_steps'] == args.steps else 'time_limit')
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    except BaseException:
        summary['status'] = 'failed'
        raise
    finally:
        summary['training_seconds'] = training_seconds
        summary['elapsed_seconds'] = time.monotonic() - started
        summary['token_presentations'] = summary['completed_steps'] * cfg.pretrain_tokens_per_step()
        summary['peak_memory_gb'] = mx.get_peak_memory() / 1024**3
        write_json(args.output / 'summary.json', summary)
        signal.signal(signal.SIGINT, old)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('config','evaluations')}), flush=True)
    return 130 if summary['status'] == 'interrupted' else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--seconds', type=int, default=180)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--independent-source-rng', action='store_true',
                        help='Keep source draws independent of corpus lengths and offset sampling')
    parser.add_argument('--optimizer', choices=['muon', 'adamw'], default='muon')
    parser.add_argument('--ns-steps', type=int, choices=[5, 10], default=10)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--mix', choices=MIXES, default='baseline')
    parser.add_argument('--scale', choices=['small', '92m', '360m'], default='small')
    parser.add_argument('--batch-size', type=int, choices=[0, 1, 2, 4, 8], default=0,
                        help='Microbatch size; 0 uses the scale default')
    parser.add_argument('--final-test', action='store_true')
    parser.add_argument('--long-context-pilot', action='store_true',
                        help='Explicit bounded 2048-context mode: at most 512 updates / 900 training seconds')
    parser.add_argument('--capabilities', action='store_true', help='Evaluate frozen BASE development tasks')
    parser.add_argument('--save-model', action='store_true', help='Save an inference-only BASE snapshot')
    parser.add_argument('--production-mix', action='store_true',
                        help='Prepare/sample all enabled corpus sources in configured token proportions')
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        return prepare(args) if args.action == 'prepare' else run(args)
    except KeyboardInterrupt:
        print('\nStopped; completed local evidence was preserved.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
