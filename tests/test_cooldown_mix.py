import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from configs.default import BEHAVIOR_NEUTRAL_ADDED_FIELDS, SpakieConfig, config_from_dict
from training.cooldown_mix import (
    CooldownMix,
    cooldown_start_step,
    previous_token_candidates,
    unlikelihood_inputs,
)

HQ = ['definitions', 'wikipedia', 'cosmopedia', 'fineweb_edu', 'arxiv', 'leads', 'stackexchange', 'finemath',
      'definitions', 'python_edu', 'cosmopedia', 'refinedweb', 'openwebmath', 'gutenberg', 'fineweb_sample', 'fineweb_edu']
V7_SLOTS = ['historical'] * 8 + [{'cycle': 'hq'}] * 6 + ['math', {'cycle': 'skill'}]


def write_mix(root: Path, seq_len: int = 8) -> Path:
    components = {}
    for name in sorted(set(HQ) | {'math', 'magnitude', 'context'}):
        tokens = np.arange(1000, dtype=np.int32) + 1000 * len(components)
        np.save(root / f'{name}.npy', tokens)
        spec = {'path': f'{name}.npy', 'natural': name in HQ}
        if name in ('math', 'magnitude', 'context'):
            np.save(root / f'{name}_starts.npy', np.arange(0, 1000, 37, dtype=np.int64))
            spec['starts'] = f'{name}_starts.npy'
        components[name] = spec
    manifest = {'slots': V7_SLOTS, 'cycles': {'hq': HQ, 'skill': ['magnitude', 'context']}, 'components': components}
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path


class CooldownMixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mix = CooldownMix.from_path(str(write_mix(self.root)), seq_len=8, seed=3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_matches_validated_v7_rows(self):
        # v7 experiment: 16 rows per microbatch, 4 microbatches per step.
        for step in range(3):
            for micro in range(4):
                for i in range(16):
                    got = self.mix.component_for_row((4 * step + micro) * 16 + i)
                    if i < 8:
                        want = 'historical'
                    elif i < 14:
                        want = HQ[(6 * (4 * step + micro) + i - 8) % len(HQ)]
                    elif i == 14:
                        want = 'math'
                    else:
                        want = ('magnitude', 'context')[(4 * step + micro) % 2]
                    self.assertEqual(got, want)

    def test_proportions_hold_for_other_batch_sizes(self):
        counts = Counter(self.mix.component_for_row(r) for r in range(8 * 6 * 64))  # B8 x accum6, 64 steps
        total = sum(counts.values())
        self.assertAlmostEqual(counts['historical'] / total, 0.5)
        self.assertAlmostEqual(counts['math'] / total, 1 / 16)
        self.assertAlmostEqual((counts['magnitude'] + counts['context']) / total, 1 / 16)
        self.assertEqual(counts['definitions'], counts['cosmopedia'])

    def test_apply_is_deterministic_keeps_historical_rows_and_snaps_curricula(self):
        x = np.zeros((16, 8), dtype=np.int32)
        y = np.ones((16, 8), dtype=np.int32)
        a = self.mix.apply(x, y, microbatch_index=5)
        b = self.mix.apply(x, y, microbatch_index=5)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        mx_, my_, natural = a
        np.testing.assert_array_equal(mx_[:8], x[:8])
        np.testing.assert_array_equal(my_[:8], y[:8])
        self.assertTrue(natural[:14].all() and not natural[14:].any())
        np.testing.assert_array_equal(mx_[8:, 1:], my_[8:, :-1])  # targets are shifted inputs
        math_offset = int(mx_[14, 0]) - 1000 * sorted(set(HQ) | {'math', 'magnitude', 'context'}).index('math')
        self.assertEqual(math_offset % 37, 0)
        self.assertFalse(np.shares_memory(mx_, x))

    def test_missing_component_rejected(self):
        manifest = json.loads((self.root / 'manifest.json').read_text())
        del manifest['components']['math']
        with self.assertRaisesRegex(ValueError, 'missing components'):
            CooldownMix(manifest, self.root, seq_len=8)

    def test_unlikelihood_inputs_mask_target_ineligible_and_non_natural(self):
        x = np.array([[5, 6, 5, 7], [5, 6, 5, 7]])
        y = np.array([[6, 5, 7, 5], [6, 5, 7, 5]])
        eligible = np.zeros(10, dtype=bool)
        eligible[[5, 6]] = True
        cands, weights = unlikelihood_inputs(x, y, np.array([True, False]), eligible, window=3)
        np.testing.assert_array_equal(previous_token_candidates(x, 3)[0, 2], [5, 6, 5])
        self.assertEqual(weights[1].sum(), 0)
        self.assertEqual(weights[0, 0].sum(), 1 / 4)        # t=0: candidate 5 (target 6)
        self.assertEqual(weights[0, 1].sum(), 1 / 4)        # t=1: candidates 6, 5; 5 is the target
        np.testing.assert_array_equal(weights[0, 3] > 0, [False, False, True])  # 7 ineligible, 5 target, 6 counts
        self.assertTrue((cands >= 0).all())

    def test_start_step_and_config_backfill(self):
        cfg = SpakieConfig()
        self.assertIsNone(cooldown_start_step(cfg))
        cfg.cooldown_mix_manifest, cfg.cooldown_mix_steps, cfg.pretrain_max_steps = 'm.json', 100, 1000
        self.assertEqual(cooldown_start_step(cfg), 900)
        cfg.cooldown_mix_start_step = 10
        self.assertEqual(cooldown_start_step(cfg), 10)
        old = asdict(SpakieConfig())
        for name in BEHAVIOR_NEUTRAL_ADDED_FIELDS:
            del old[name]
        restored = config_from_dict(old)
        self.assertEqual(restored.cooldown_mix_manifest, '')
        self.assertEqual(restored.cooldown_ul_alpha, 0.0)
        with self.assertRaisesRegex(ValueError, 'unknown'):
            config_from_dict({**asdict(SpakieConfig()), 'bogus': 1})


class UnlikelihoodParityTests(unittest.TestCase):
    def test_mlx_and_torch_penalties_match(self):
        try:
            import mlx.core as mx
            import torch
        except ImportError:
            self.skipTest('needs both backends')
        from training.pretrain import unlikelihood_penalty_torch
        from training.pretrain_mlx import unlikelihood_penalty_mlx
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(2, 6, 11)).astype(np.float32)
        x = rng.integers(0, 11, size=(2, 6))
        y = rng.integers(0, 11, size=(2, 6))
        cands, weights = unlikelihood_inputs(x, y, np.array([True, True]), np.ones(11, dtype=bool), window=3)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        expected = float((-np.log(1 - np.exp(np.take_along_axis(logp, cands, -1))) * weights).sum())
        got_mlx = float(unlikelihood_penalty_mlx(mx.array(logp), mx.array(cands), mx.array(weights)).item())
        got_torch = float(unlikelihood_penalty_torch(torch.from_numpy(logp), torch.from_numpy(cands).long(),
                                                     torch.from_numpy(weights)))
        self.assertAlmostEqual(got_mlx, expected, places=5)
        self.assertAlmostEqual(got_torch, expected, places=5)


if __name__ == '__main__':
    unittest.main()


class CooldownMixSafetyTests(unittest.TestCase):
    """Regression tests for the review findings on the cooldown-mix path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path = write_mix(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self):
        return json.loads(self.manifest_path.read_text())

    def test_tokenizer_hash_and_token_range_are_checked(self):
        tokenizer = self.root / 'tok.model'
        tokenizer.write_bytes(b'tokenizer-a')
        manifest = self._manifest()
        manifest['tokenizer_sha256'] = 'not-the-hash'
        with self.assertRaisesRegex(ValueError, 'built for tokenizer'):
            CooldownMix(manifest, self.root, seq_len=8, tokenizer_path=str(tokenizer))
        from training.cooldown_mix import sha256_file
        manifest['tokenizer_sha256'] = sha256_file(tokenizer)
        CooldownMix(manifest, self.root, seq_len=8, tokenizer_path=str(tokenizer), vocab_size=100_000)
        with self.assertRaisesRegex(ValueError, 'vocab size'):
            CooldownMix(manifest, self.root, seq_len=8, tokenizer_path=str(tokenizer), vocab_size=500)

    def test_changed_document_starts_fail_verification_and_change_fingerprint(self):
        from training.cooldown_mix import sha256_file
        manifest = self._manifest()
        for spec in manifest['components'].values():
            spec['sha256'] = sha256_file(self.root / spec['path'])
            if spec.get('starts'):
                spec['starts_sha256'] = sha256_file(self.root / spec['starts'])
        before = CooldownMix(manifest, self.root, seq_len=8, verify=True).fingerprint()
        np.save(self.root / 'math_starts.npy', np.arange(0, 1000, 41, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, 'changed'):
            CooldownMix(manifest, self.root, seq_len=8, verify=True)
        manifest['components']['math']['starts_sha256'] = sha256_file(self.root / 'math_starts.npy')
        self.assertNotEqual(CooldownMix(manifest, self.root, seq_len=8, verify=True).fingerprint(), before)

    def test_resume_refuses_a_different_mix(self):
        from training.cooldown_mix import load_cooldown_mix_for_training, sha256_file
        tokenizer = self.root / 'tok.model'
        tokenizer.write_bytes(b'tok')
        manifest = self._manifest()
        manifest['tokenizer_sha256'] = sha256_file(tokenizer)
        for spec in manifest['components'].values():
            spec['sha256'] = sha256_file(self.root / spec['path'])
            if spec.get('starts'):
                spec['starts_sha256'] = sha256_file(self.root / spec['starts'])
        self.manifest_path.write_text(json.dumps(manifest))
        cfg = SpakieConfig()
        cfg.cooldown_mix_manifest, cfg.max_seq_len = str(self.manifest_path), 8
        cfg.tokenizer_prefix, cfg.vocab_size = str(self.root / 'tok'), 100_000
        load_cooldown_mix_for_training(cfg)
        pinned = cfg.cooldown_mix_fingerprint
        self.assertTrue(pinned)
        load_cooldown_mix_for_training(cfg)  # same inputs resume fine
        cfg.cooldown_mix_seed = 1
        with self.assertRaisesRegex(ValueError, 'differ'):
            load_cooldown_mix_for_training(cfg)


class BuilderSafetyTests(unittest.TestCase):
    def test_magnitude_curriculum_honours_exclusions(self):
        from scripts.build_cooldown_mix import curricula
        from scripts.build_magnitude_curriculum import build
        first = build(16384, 7, symmetric=True, rich=True)
        pair = next(d['meta']['pair'] for d in first if 'pair' in d['meta'])
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / 'tasks.json'
            tasks.write_text(json.dumps([{'values': pair, 'entities': ['Arlo']}]))
            texts = curricula(7, [tasks])['magnitude']
        a, b = pair
        self.assertFalse(any(f'{a} and {b}' in t or f'{b} and {a}' in t or f'{a} or {b}' in t or f'{b} or {a}' in t
                             for t in texts))
        self.assertFalse(any('Arlo' in t for t in texts))

    def _synthetic_split(self, root, *, train_fraction, train_target):
        from scripts.prepare_data import AcceptedDocument, SHARD_RESUME_JOURNAL, append_accepted_document, merge_shards
        rng = np.random.default_rng(1)
        docs = [(('a', 'b', 'b', 'c')[i % 4] if i < 40 else 'a', i + 1, int(rng.integers(3, 40))) for i in range(60)]
        shard_dir = root / 'shards'
        shard_dir.mkdir()
        with (shard_dir / SHARD_RESUME_JOURNAL).open('wb') as handle:
            for source, doc_id, length in docs:
                append_accepted_document(handle, AcceptedDocument(source, length, length, doc_id, ()))
        tokens = np.concatenate([np.full(length, doc_id, dtype=np.uint16) for _, doc_id, length in docs])
        cut = len(tokens) // 3
        shards = [shard_dir / 'tokens-0.npy', shard_dir / 'tokens-1.npy']
        np.save(shards[0], tokens[:cut])
        np.save(shards[1], tokens[cut:])
        runs, ends, totals, cursor = [], {}, {}, 0
        for source, _, length in docs:
            if runs and runs[-1][0] == source:
                runs[-1] = (source, runs[-1][1], cursor + length)
            else:
                runs.append((source, cursor, cursor + length))
            cursor += length
            totals[source] = totals.get(source, 0) + length
            ends.setdefault(source, []).append(totals[source])
        train_tokens, _ = merge_shards(shards, root / 'train.npy', root / 'val.npy', train_fraction, np.uint16,
                                       train_tokens_target=train_target, source_runs=runs, source_document_ends=ends)
        settings = {'train_split_fraction': train_fraction, 'target_train_tokens': train_target,
                    'train_tokens': train_tokens, 'near_dup_jaccard_threshold': 0.8, 'near_dup_num_perm': 128,
                    'near_dup_shingle_size': 5}
        return shard_dir, settings, set(np.unique(np.load(root / 'train.npy')).tolist()), \
            set(np.unique(np.load(root / 'val.npy')).tolist())

    def test_train_membership_uses_recorded_non_default_split(self):
        from scripts.build_cooldown_mix import split_document_hashes
        with tempfile.TemporaryDirectory() as tmp:
            # A non-default token target, as the review's reproduction used.
            shard_dir, settings, in_train, in_val = self._synthetic_split(Path(tmp), train_fraction=0.7, train_target=400)
            train_hashes, val_hashes, _, _ = split_document_hashes(shard_dir, settings)
            with self.assertRaisesRegex(ValueError, 'reproduces the committed'):
                split_document_hashes(shard_dir, {**settings, 'target_train_tokens_candidates': [0, 123],
                                                  'train_split_fraction': 0.95})
            # A stale report target is skipped in favour of one that reproduces train.npy.
            stale = {**settings, 'target_train_tokens_candidates': [999, settings['target_train_tokens']]}
            split_document_hashes(shard_dir, stale)
            self.assertEqual(stale['target_train_tokens'], 400)
        selected = {int(h) for values in train_hashes.values() for h in values}
        held_out = {int(h) for values in val_hashes.values() for h in values}
        self.assertEqual(selected, in_train)
        self.assertEqual(held_out, in_val)
        self.assertTrue(in_val)

    def test_validation_near_duplicates_are_detected(self):
        from scripts.prepare_data import NearDuplicateIndex, compute_minhash_signature
        base = ' '.join(f'word{i} alpha beta gamma delta' for i in range(300))
        near = base + ' a different final paragraph with a few new words'
        other = ' '.join(f'token{i} unrelated text here' for i in range(300))
        index = NearDuplicateIndex(threshold=0.8, num_perm=128, shingle_size=5)
        sig = lambda text: compute_minhash_signature(text, num_perm=128, shingle_size=5)
        index.insert_known(1, index.band_keys(sig(base)))
        self.assertTrue(index.collides(sig(near)))
        self.assertFalse(index.collides(sig(other)))
        self.assertFalse(index.collides(sig(other)))  # read-only: checking never inserts


class RecordedSplitTests(unittest.TestCase):
    def test_merge_records_split_and_builder_prefers_it_over_the_report(self):
        from scripts.build_cooldown_mix import recorded_split_settings
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir, settings, _, _ = BuilderSafetyTests()._synthetic_split(root, train_fraction=0.7, train_target=400)
            manifest = json.loads((root / 'processed_data_manifest.json').read_text())
            self.assertEqual(manifest['split'], {'train_split_fraction': 0.7, 'train_tokens_target': 400})
            manifest['preparation'] = {'config': {'train_split_fraction': 0.7, 'near_dup_jaccard_threshold': 0.8,
                                                  'near_dup_num_perm': 128, 'near_dup_shingle_size': 5}}
            (root / 'processed_data_manifest.json').write_text(json.dumps(manifest))
            (root / 'corpus_report.json').write_text(json.dumps({'target_train_tokens': 999}))  # stale dry-run
            recorded = recorded_split_settings(root)
        self.assertEqual(recorded['target_train_tokens_candidates'], [400])
        self.assertTrue(recorded['split_recorded_in_manifest'])


class UnlikelihoodNormalizationTests(unittest.TestCase):
    def test_weights_do_not_depend_on_microbatch_partitioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            mix = CooldownMix.from_path(str(write_mix(Path(tmp))), seq_len=8, seed=3)
            rng = np.random.default_rng(0)
            x = rng.integers(0, 50, size=(16, 8))
            y = rng.integers(0, 50, size=(16, 8))
            eligible = np.ones(50, dtype=bool)
            natural = np.array([mix.is_natural(mix.component_for_row(r)) for r in range(16)])
            n = mix.natural_rows_in_step(0, 16, 1)
            self.assertEqual(n, mix.natural_rows_in_step(0, 8, 2))
            _, whole = unlikelihood_inputs(x, y, natural, eligible, 4, natural_rows=n)
            _, first = unlikelihood_inputs(x[:8], y[:8], natural[:8], eligible, 4, natural_rows=n)
            _, second = unlikelihood_inputs(x[8:], y[8:], natural[8:], eligible, 4, natural_rows=n)
            np.testing.assert_allclose(np.concatenate([first, second]), whole)
            # Per-microbatch normalization (the old behaviour) would weight the halves differently.
            _, old_first = unlikelihood_inputs(x[:8], y[:8], natural[:8], eligible, 4)
            _, old_second = unlikelihood_inputs(x[8:], y[8:], natural[8:], eligible, 4)
            self.assertNotAlmostEqual(float(old_first.max()), float(old_second.max()))


if __name__ == '__main__':
    unittest.main()
