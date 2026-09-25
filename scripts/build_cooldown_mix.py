"""Build the anneal-v7 cooldown mix (components + manifest) for training/cooldown_mix.py.

Validated on the 94M checkpoint as a cooldown-phase change (two data-draw seeds,
untouched confirmation): half of each batch stays on the main corpus; the rest is
a quality cycle over every source plus Wikipedia definitions/leads, verified
worked addition, and alternating pairwise-magnitude / context-retrieval
documents.

Split membership is read from prepare_data's accepted-document journal, using the
split settings recorded with the processed arrays, and the recomputed train total
must equal the committed train.npy size. Source samples must be documents placed in
train.npy. Wikipedia leads/definitions may come from any article (prepare_data stops
reading a source once its budget is met) except validation articles and their MinHash
near-duplicates, checked with prepare_data's own LSH banding.
Run this after prepare_data has finished.
"""
from __future__ import annotations

from array import array
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = ('wikipedia_snapshot', 'cosmopedia_v2', 'fineweb-edu', 'arxiv', 'stackexchange', 'finemath',
           'python_edu', 'refinedweb', 'openwebmath', 'gutenberg', 'fineweb_sample')
HQ_CYCLE = ['wiki_definitions', 'wikipedia_snapshot', 'cosmopedia_v2', 'fineweb-edu', 'arxiv', 'wiki_leads',
            'stackexchange', 'finemath', 'wiki_definitions', 'python_edu', 'cosmopedia_v2', 'refinedweb',
            'openwebmath', 'gutenberg', 'fineweb_sample', 'fineweb-edu']
SLOTS = ['historical'] * 8 + [{'cycle': 'hq'}] * 6 + ['math', {'cycle': 'skill'}]
SKILL_CYCLE = ['magnitude', 'context']


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_split_settings(processed_dir: Path) -> dict:
    """Split and dedup settings that produced processed_dir, read from its committed manifest.

    Manifests published before the "split" record existed carry no train-token target;
    split_document_hashes then accepts only a candidate target that exactly reproduces
    the committed train.npy size.
    """
    manifest = json.loads((processed_dir / 'processed_data_manifest.json').read_text())
    prep = (manifest.get('preparation') or {}).get('config') or {}
    missing = [k for k in ('train_split_fraction', 'near_dup_jaccard_threshold', 'near_dup_num_perm',
                           'near_dup_shingle_size') if k not in prep]
    if missing:
        raise ValueError(f'processed data lacks recorded preparation settings: {missing}')
    split = manifest.get('split') or {}
    if split:
        candidates = [int(split['train_tokens_target'])]
    else:
        from configs.default import SpakieConfig
        candidates = [int(SpakieConfig().target_train_tokens), 0]
        report = processed_dir / 'corpus_report.json'
        if report.is_file():
            candidates.insert(0, int(json.loads(report.read_text()).get('target_train_tokens', 0)))
    return {'train_split_fraction': float(split.get('train_split_fraction', prep['train_split_fraction'])),
            'target_train_tokens_candidates': list(dict.fromkeys(candidates)),
            'split_recorded_in_manifest': bool(split),
            'train_tokens': int(manifest['train']['tokens']),
            'near_dup_jaccard_threshold': float(prep['near_dup_jaccard_threshold']),
            'near_dup_num_perm': int(prep['near_dup_num_perm']),
            'near_dup_shingle_size': int(prep['near_dup_shingle_size'])}


def split_document_hashes(shard_dir: Path, settings: dict, near_dup_sources: tuple[str, ...] = ()):
    """Per-source sorted exact hashes of train.npy and val.npy documents, plus a near-duplicate
    index of validation documents for near_dup_sources."""
    from scripts.prepare_data import (NearDuplicateIndex, SHARD_RESUME_JOURNAL, compute_source_train_targets,
                                      iter_accepted_documents)
    journal = shard_dir / SHARD_RESUME_JOURNAL
    if not journal.is_file():
        raise FileNotFoundError(f"prepare_data journal not found: {journal}; run prepare_data first")
    # Two streaming passes keep memory to one uint64 per document, as prepare_data does.
    totals: dict[str, int] = {}
    ends: dict[str, array] = {}
    for record in iter_accepted_documents(journal):
        totals[record.source] = totals.get(record.source, 0) + record.token_count
        ends.setdefault(record.source, array('Q')).append(totals[record.source])
    candidates = settings.get('target_train_tokens_candidates', [settings.get('target_train_tokens', 0)])
    for target in candidates:
        targets, split_idx = compute_source_train_targets(totals, sum(totals.values()),
                                                          settings['train_split_fraction'], target, ends)
        if split_idx == settings['train_tokens']:
            settings['target_train_tokens'] = target
            break
    else:
        raise ValueError(f"no recorded split setting reproduces the committed train.npy "
                         f"({settings['train_tokens']:,} tokens; tried targets {candidates}); "
                         "journal and arrays are out of sync")
    del ends
    seen: dict[str, int] = {}
    train: dict[str, array] = {}
    val: dict[str, array] = {}
    val_index = NearDuplicateIndex(threshold=settings['near_dup_jaccard_threshold'],
                                   num_perm=settings['near_dup_num_perm'],
                                   shingle_size=settings['near_dup_shingle_size'])
    for record in iter_accepted_documents(journal):
        seen[record.source] = seen.get(record.source, 0) + record.token_count
        in_train_split = seen[record.source] <= targets[record.source]
        split = train if in_train_split else val
        split.setdefault(record.source, array('Q')).append(record.exact_hash)
        if not in_train_split and record.source in near_dup_sources:
            if not record.band_keys:
                raise ValueError('journal has no MinHash band keys; cannot exclude validation near-duplicates')
            val_index.insert_known(record.exact_hash, record.band_keys)
    info = {'journal': str(journal), 'journal_sha256': sha(journal), 'train_targets': targets,
            'train_documents': {k: len(v) for k, v in train.items()},
            'val_documents': {k: len(v) for k, v in val.items()}}
    as_sorted = lambda split: {k: np.unique(np.frombuffer(v, dtype=np.uint64)) for k, v in split.items()}
    return as_sorted(train), as_sorted(val), val_index, info


def in_split(hashes: dict[str, np.ndarray], source: str, cleaned_text: str) -> bool:
    from scripts.prepare_data import compute_exact_hash
    values = hashes.get(source)
    if values is None or not len(values):
        return False
    key = np.uint64(compute_exact_hash(cleaned_text))
    index = int(np.searchsorted(values, key))
    return index < len(values) and values[index] == key


def raw_shards(raw_root: Path, source: str) -> list[Path]:
    # Same ordering as prepare_data.iter_input_files (sorted resolved paths).
    return sorted(p.resolve() for p in (raw_root / source).rglob('*.jsonl'))


def source_sample(raw_root: Path, source: str, tokenizer, token_budget: int,
                  train_hashes: dict[str, np.ndarray]) -> list[list[int]]:
    from scripts.prepare_data import clean_text
    docs, total = [], 0
    for path in raw_shards(raw_root, source):
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                text = clean_text(json.loads(line).get('text', ''), source)
                if not in_split(train_hashes, source, text):
                    continue
                ids = tokenizer.encode(text, add_eos=True)
                docs.append(ids)
                total += len(ids)
                if total >= token_budget:
                    return docs
    return docs


def wikipedia_leads_and_definitions(raw_root: Path, tokenizer, val_hashes: dict[str, np.ndarray], val_index,
                                    settings: dict, max_shards: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    """One pass over Wikipedia articles, skipping validation articles and their near-duplicates.

    leads: title + text before the first section break, articles > 40k chars.
    definitions: title + first two sentences, articles > 15k chars.
    """
    from scripts.prepare_data import clean_text, compute_minhash_signature
    shards = raw_shards(raw_root, 'wikipedia_snapshot')
    leads, definitions = [], []
    for path in shards[:max_shards] if max_shards > 0 else shards:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get('text', '')
                if len(text) < 15000:
                    continue
                cleaned = clean_text(text, 'wikipedia_snapshot')
                if in_split(val_hashes, 'wikipedia_snapshot', cleaned) or val_index.collides(
                        compute_minhash_signature(cleaned, num_perm=settings['near_dup_num_perm'],
                                                  shingle_size=settings['near_dup_shingle_size'])):
                    continue
                title = row.get('title', '')
                first = re.split(r'\n', text, maxsplit=1)[0].strip()
                definition = ' '.join(re.split(r'(?<=[a-z0-9)]\.) (?=[A-Z])', first)[:2])
                if 60 <= len(definition) <= 600:
                    definitions.append(tokenizer.encode(title + '\n' + definition, add_eos=True))
                if len(text) >= 40000:
                    lead = re.split(r'\n\s*\n|\n[A-Z][^\n]{0,60} ?\n', text, maxsplit=1)[0].strip()
                    if 200 <= len(lead) <= 2000 and lead.count('\n') <= 3:
                        leads.append(tokenizer.encode(title + '\n' + lead, add_eos=True))
    return leads, definitions


def curricula(seed: int, exclude_tasks: list[Path] = ()) -> dict[str, list[str]]:
    from scripts.build_reasoning_curriculum import build as build_reasoning
    from scripts.build_magnitude_curriculum import banned_from_tasks, build as build_magnitude
    from scripts.eval_reasoning_probe import reasoning_cases
    documents, _, _ = build_reasoning(decomposed_addition=True, broad_numbers=True)
    dev_pairs = set()
    for task in reasoning_cases('dev'):
        for a, b, _ in task['expected_equations']:
            dev_pairs.add(('+', min(a, b), max(a, b)))
    math = [d['text'] for d in documents if d['kind'] == 'addition_explanation'
            and not any((op, min(a, b), max(a, b)) in dev_pairs for op, a, b in d['pairs'])]
    context = [d['text'] for d in documents if d.get('kind') in ('retrieve', 'maximum', 'minimum')]
    banned_pairs, banned_names = banned_from_tasks(exclude_tasks)
    magnitude = [d['text'] for d in build_magnitude(16384, seed, banned_pairs, banned_names, symmetric=True, rich=True)]
    return {'math': math, 'context': context, 'magnitude': magnitude}


def write_component(out: Path, name: str, docs: list[list[int]], *, natural: bool, with_starts: bool) -> dict:
    lengths = np.fromiter((len(d) for d in docs), dtype=np.int64, count=len(docs))
    tokens = np.fromiter((t for d in docs for t in d), dtype=np.uint16, count=int(lengths.sum()))
    np.save(out / f'{name}.npy', tokens)
    spec = {'path': f'{name}.npy', 'natural': natural, 'documents': len(docs), 'tokens': int(len(tokens)),
            'sha256': sha(out / f'{name}.npy')}
    if with_starts:
        np.save(out / f'{name}_starts.npy', np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
        spec['starts'] = f'{name}_starts.npy'
        spec['starts_sha256'] = sha(out / f'{name}_starts.npy')
    return spec


def main():
    from configs.default import SpakieConfig
    from tokenizer.train_tokenizer import SpakieTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw/large_corpus')
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    parser.add_argument('--tokens-per-source', type=int, default=4_000_000)
    parser.add_argument('--seed', type=int, default=2026092403)
    parser.add_argument('--max-wiki-shards', type=int, default=0, help='Limit Wikipedia shards scanned (smoke tests only)')
    parser.add_argument('--processed-dir', type=Path, default=None,
                        help='prepare_data output with train.npy, its manifest and shards/ (default: config)')
    parser.add_argument('--exclude-tasks', type=Path, nargs='*', default=[],
                        help='Evaluation task JSON files whose number pairs / entity names the magnitude curriculum must avoid')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new directory')
    config = SpakieConfig()
    tokenizer = SpakieTokenizer(str(args.tokenizer))
    processed_dir = args.processed_dir or Path(config.processed_data_dir)
    settings = recorded_split_settings(processed_dir)
    train_hashes, val_hashes, val_index, membership = split_document_hashes(
        processed_dir / 'shards', settings, near_dup_sources=('wikipedia_snapshot',))
    membership['recorded_settings'] = settings
    args.output.mkdir(parents=True)
    components = {}
    for source in SOURCES:
        docs = source_sample(args.raw_dir, source, tokenizer, args.tokens_per_source, train_hashes)
        components[source] = write_component(args.output, source, docs, natural=True, with_starts=False)
        print(source, components[source]['tokens'], flush=True)
    leads, definitions = wikipedia_leads_and_definitions(args.raw_dir, tokenizer, val_hashes, val_index, settings,
                                                         args.max_wiki_shards)
    components['wiki_leads'] = write_component(args.output, 'wiki_leads', leads, natural=True, with_starts=False)
    components['wiki_definitions'] = write_component(args.output, 'wiki_definitions', definitions,
                                                     natural=True, with_starts=False)
    for name, texts in curricula(args.seed, args.exclude_tasks).items():
        docs = [tokenizer.encode(t, add_eos=True) for t in texts]
        components[name] = write_component(args.output, name, docs, natural=False, with_starts=True)
        print(name, components[name]['documents'], components[name]['tokens'], flush=True)
    manifest = {'recipe': 'anneal_v7', 'slots': SLOTS, 'cycles': {'hq': HQ_CYCLE, 'skill': SKILL_CYCLE},
                'components': components, 'tokenizer_sha256': sha(args.tokenizer),
                'builder_sha256': sha(Path(__file__)), 'seed': args.seed,
                'train_membership': membership,
                'exclude_tasks_sha256': {str(p): sha(p) for p in args.exclude_tasks},
                'recommended': {'cooldown_mix_steps_for_98k_token_updates': 512, 'cooldown_ul_alpha': 0.25,
                                'note': '512 x 98,304 tokens matches the ~50M-token validated cooldown exposure.'}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v['tokens'] for k, v in components.items()}))


if __name__ == '__main__':
    main()
