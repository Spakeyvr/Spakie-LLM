import copy
import unittest

import numpy as np

from scripts.eval_repeat_order import (aggregate_records, build_protocol,
                                      summarize_prediction_rows, validate_protocol)


class RepeatOrderTests(unittest.TestCase):
    def test_protocol_pairs_preserve_tokens_but_exclude_original_transitions(self):
        shared = {'▁aa', '▁bb', '▁cc', '▁dd', '▁ee', '▁ff', '▁gg', '▁hh'}
        inputs = [shared | {'▁onlyleft', '<pad>'}, shared | {'▁onlyright', '▁123'}]
        protocol = build_protocol(inputs, pool_size=8, lengths=(3, 8), cases_per_length=12)
        self.assertEqual(protocol, build_protocol(inputs, pool_size=8, lengths=(3, 8), cases_per_length=12))
        self.assertEqual(set(protocol['pool_pieces']), shared)
        validate_protocol(protocol)
        for row in protocol['cases']:
            self.assertEqual(sorted(row['repeat'][1:]), sorted(row['shuffled'][1:]))
            self.assertEqual(row['repeat'][0], row['shuffled'][0])
            self.assertTrue(set(zip(row['repeat'], row['repeat'][1:])).isdisjoint(
                zip(row['shuffled'], row['shuffled'][1:])))

    def test_impossible_protocols_fail_instead_of_looping(self):
        words = [{'▁aa', '▁bb', '▁cc'}]
        for kwargs in ({'lengths':(2,), 'pool_size':3},
                       {'lengths':(3,), 'pool_size':4},
                       {'lengths':(3, 3), 'pool_size':3},
                       {'lengths':(3,), 'pool_size':3, 'cases_per_length':0}):
            with self.assertRaises(ValueError):
                build_protocol(words, **kwargs)
        good = build_protocol(words, pool_size=3, lengths=(3,), cases_per_length=1)
        bad = copy.deepcopy(good)
        bad['cases'][0]['shuffled'] = bad['cases'][0]['first']
        with self.assertRaisesRegex(ValueError, 'adjacent'):
            validate_protocol(bad)
        bad = copy.deepcopy(good)
        bad['cases'][0]['shuffled'][-1] = 999
        with self.assertRaisesRegex(ValueError, 'multiset'):
            validate_protocol(bad)

    def test_scoring_excludes_both_unpredictable_boundary_targets(self):
        # EOS+A+B gives six target rows at L=3; only positions1,2 and4,5 count.
        nll = np.array([[999., 2., 4., 888., .2, .4],
                        [999., 2., 4., 888., 5., 7.]])
        correct = np.array([[True, False, True, True, True, True],
                            [True, False, True, True, False, False]])
        row = summarize_prediction_rows(nll, correct, 3)
        self.assertEqual(row['scored_tokens'], 2)
        self.assertEqual(row['first_nll'], 3.)
        self.assertAlmostEqual(row['repeat_nll'], .3)
        self.assertEqual(row['shuffle_nll'], 6.)
        self.assertEqual((row['first_correct'], row['repeat_correct'], row['shuffle_correct']), (1, 2, 0))
        self.assertEqual(row['first_condition_max_nll_delta'], 0.)
        metrics = aggregate_records([dict(row, length=3)])['3']
        self.assertEqual(metrics['repeat_accuracy'], 1.)
        self.assertEqual(metrics['shuffle_accuracy'], 0.)
        self.assertAlmostEqual(metrics['shuffle_minus_repeat_nll'], 5.7)
        with self.assertRaisesRegex(ValueError, 'complete sequences'):
            summarize_prediction_rows(nll[:, :-1], correct[:, :-1], 3)
        nll[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            summarize_prediction_rows(nll, correct, 3)


if __name__ == '__main__':
    unittest.main()
