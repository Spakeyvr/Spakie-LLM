import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import eval_fact_confidence as fc


def lse(row):
    return math.log(sum(math.exp(v) for v in row))


class CanonicalAnswerTests(unittest.TestCase):
    def test_covers_all_24_facts_and_matches_greedy_regex(self):
        cases = fc.fact_cases()
        self.assertEqual(len(cases), 24)
        for case in cases:
            self.assertRegex(case['canonical_answer'], re.compile(case['expected_regex']))

    def test_module_imports_without_mlx(self):
        code = ('import sys; import scripts.eval_fact_confidence as m; m.fact_cases(); '
                'assert not any(n == "mlx" or n.startswith("mlx.") for n in sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).resolve().parents[1])


class AlignmentTests(unittest.TestCase):
    def test_returns_answer_suffix(self):
        self.assertEqual(fc.answer_ids_from_alignment([5, 6], [5, 6, 7, 8]), [7, 8])

    def test_rejects_non_prefix_merge(self):
        with self.assertRaisesRegex(ValueError, 'not an exact prefix'):
            fc.answer_ids_from_alignment([5, 6], [5, 9, 7], fact_id='fact_x')

    def test_rejects_empty_answer_and_prompt(self):
        with self.assertRaisesRegex(ValueError, 'adds no tokens'):
            fc.answer_ids_from_alignment([5, 6], [5, 6])
        with self.assertRaisesRegex(ValueError, 'no tokens'):
            fc.answer_ids_from_alignment([], [1])

    def test_logit_rows_use_causal_positions(self):
        # Input is combined[:-1] = 2 prompt + 2 answer - 1 = 3 rows; row k predicts token k+1.
        logits = np.arange(12, dtype=np.float32).reshape(3, 4)
        rows = fc.answer_logit_rows(logits, prompt_len=2, n_answer=2)
        np.testing.assert_array_equal(rows, logits[1:])
        with self.assertRaisesRegex(ValueError, 'logit rows'):
            fc.answer_logit_rows(logits, prompt_len=2, n_answer=3)


class TokenStatTests(unittest.TestCase):
    rows = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])

    def test_hand_computed_multi_token_answer(self):
        ref = [0, 2]
        rivals = fc.select_rivals(self.rows, ref)
        self.assertEqual(rivals, [1, 1])
        stats = fc.token_stats(self.rows, ref, rivals)
        r0, r1 = self.rows
        exp_ref = [2 - lse(r0), 1 - lse(r1)]
        exp_margin = [2 - 1, 1 - 3]
        for pos, a, m in zip(stats['positions'], exp_ref, exp_margin):
            self.assertAlmostEqual(pos['ref_logprob'], a)
            self.assertAlmostEqual(pos['margin'], m)
        self.assertAlmostEqual(stats['sum_ref_logprob'], sum(exp_ref))
        self.assertAlmostEqual(stats['mean_ref_logprob'], sum(exp_ref) / 2)
        self.assertAlmostEqual(stats['mean_margin'], -0.5)
        self.assertAlmostEqual(stats['min_margin'], -2.0)
        self.assertEqual([p['ref_is_argmax'] for p in stats['positions']], [True, False])
        self.assertFalse(stats['all_ref_argmax'])

    def test_later_divergence_is_visible(self):
        # "100 degrees Celsius" vs "Fahrenheit": first token confident, last is not.
        rows = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 4.0]])
        stats = fc.token_stats(rows, [0, 1, 0], fc.select_rivals(rows, [0, 1, 0]))
        self.assertGreater(stats['positions'][0]['margin'], 0)
        self.assertLess(stats['min_margin'], 0)

    def test_offset_invariance(self):
        ref = [0, 2]
        rivals = fc.select_rivals(self.rows, ref)
        shifted = self.rows + np.array([[1000.0], [-37.5]])
        self.assertEqual(fc.select_rivals(shifted, ref), rivals)
        a, b = fc.token_stats(self.rows, ref, rivals), fc.token_stats(shifted, ref, rivals)
        for k in fc.SUMMARY_KEYS:
            self.assertAlmostEqual(a[k], b[k], places=9)

    def test_parent_rivals_stay_frozen_for_candidate(self):
        ref = [0, 2]
        parent_rivals = fc.select_rivals(self.rows, ref)
        candidate = np.array([[2.0, 0.0, 9.0], [4.0, 0.0, 1.0]])  # candidate's own rivals differ
        self.assertNotEqual(fc.select_rivals(candidate, ref), parent_rivals)
        stats = fc.token_stats(candidate, ref, parent_rivals)
        lp = candidate - np.array([[lse(r)] for r in candidate])
        self.assertAlmostEqual(stats['positions'][0]['rival_logprob'], lp[0, 1])
        self.assertAlmostEqual(stats['positions'][1]['rival_logprob'], lp[1, 1])

    def test_rejects_rival_equal_to_reference_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            fc.token_stats(self.rows, [0, 2], [0, 1])
        with self.assertRaises(ValueError):
            fc.token_stats(self.rows, [0], [1])

    def test_changes_and_aggregate(self):
        ref = [0, 2]
        rivals = fc.select_rivals(self.rows, ref)
        parent = fc.token_stats(self.rows, ref, rivals)
        cand = fc.token_stats(self.rows + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]), ref, rivals)
        changes = fc.stat_changes(parent, cand)
        self.assertAlmostEqual(changes['delta_positions'][0]['margin'], 0.0)
        self.assertAlmostEqual(changes['delta_positions'][1]['margin'], 3.0)
        self.assertAlmostEqual(changes['delta_mean_margin'], 1.5)
        agg = fc.aggregate([dict(stats=cand, changes=changes)])
        self.assertEqual(agg['n_facts'], 1)
        self.assertEqual(agg['n_min_margin_positive'], 1)
        self.assertAlmostEqual(agg['mean_delta_min_margin'], cand['min_margin'] - parent['min_margin'])
        self.assertNotIn('mean_delta_min_margin', fc.aggregate([dict(stats=parent)]))


class MetadataTests(unittest.TestCase):
    def meta(self, **config):
        base = dict(vocab_size=8, n_layers=2, d_model=4, pretrain_lr=1e-3)
        return dict(stage='pretrain', config=dict(base, **config), tokenizer={'sha256': 'a'})

    def test_accepts_training_only_differences(self):
        fc.check_metadata(self.meta(), self.meta(pretrain_lr=5e-4), source='c')

    def test_rejects_stage_model_and_tokenizer_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'stage'):
            fc.check_metadata(self.meta(), dict(self.meta(), stage='sft'), source='c')
        with self.assertRaisesRegex(ValueError, 'stage'):
            fc.check_metadata(self.meta(), {k: v for k, v in self.meta().items() if k != 'stage'}, source='c')
        with self.assertRaisesRegex(ValueError, 'n_layers'):
            fc.check_metadata(self.meta(), self.meta(n_layers=3), source='c')
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            fc.check_metadata(self.meta(), dict(self.meta(), tokenizer={'sha256': 'b'}), source='c')

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sub' / 'out.json'
            fc.create_output(path)
            with self.assertRaises(FileExistsError):
                fc.create_output(path)
            fc.write_json(path, {'ok': 1})
            self.assertEqual(path.read_text().strip().replace(' ', '').replace('\n', ''), '{"ok":1}')


if __name__ == '__main__':
    unittest.main()
