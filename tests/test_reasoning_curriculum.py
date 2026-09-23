import unittest

from scripts.build_reasoning_curriculum import (build, addition_explanations,
    score_reasoned_completion, TRAIN_NAMES, DEV_NAMES, TEST_NAMES)


class ReasoningCurriculumTests(unittest.TestCase):
    def test_addition_audit_checks_overlapping_claims_in_chains(self):
        from scripts.build_reasoning_curriculum import addition_claim_diagnostics
        for text, invalid in [('1 + 2 = 3 + 4 = 9', '3 + 4 = 9'),
                              ('1 plus 2 equals 3 plus 4 equals 9', '3 plus 4 equals 9')]:
            result = addition_claim_diagnostics(text)
            self.assertEqual(result['recognized_addition_claims'], 2)
            self.assertEqual(len(result['invalid_addition_claims']), 1)
            claim = result['invalid_addition_claims'][0]
            self.assertEqual(claim['claim'], invalid)
            self.assertEqual(text[claim['start']:claim['end']], invalid)

    def test_language_validation_respects_pilot_context_and_preserves_old_windows(self):
        import numpy as np
        from scripts.eval_reasoning_probe import validation_windows
        for context in (256, 2048, 4096):
            length = min(2048, context)
            token_count = 200 * length + 1
            seq_len, dev = validation_windows(token_count, context, 17, 'dev')
            _, test = validation_windows(token_count, context, 17, 'test')
            self.assertEqual(seq_len, length)
            self.assertEqual((len(dev), len(test)), (32, 96))
            self.assertTrue(set(dev).isdisjoint(test))
            self.assertTrue(all((int(i) + 1) * seq_len < token_count for i in [*dev, *test]))
            expected = np.random.default_rng(17).choice(200, size=128, replace=False)
            np.testing.assert_array_equal(np.concatenate([dev, test]), expected)

    def test_language_validation_rejects_short_arrays_without_reusing_windows(self):
        from scripts.eval_reasoning_probe import validation_windows
        for context in (256, 2048):
            with self.assertRaisesRegex(ValueError, 'needs at least'):
                validation_windows(128 * context, context, 17, 'dev')
            length, indices = validation_windows(128 * context + 1, context, 17, 'dev')
            self.assertEqual(length, context)
            self.assertEqual(len(set(indices)), 32)
        with self.assertRaisesRegex(ValueError, 'positive'):
            validation_windows(1000, 0, 17, 'dev')

    def test_sealed_pairs_remain_excluded_when_training_seed_changes(self):
        from scripts.eval_reasoning_probe import SEALED_PAIRS
        from scripts.probe_base_learning import pair_key
        held = {pair_key('+', a, b) for a,b in SEALED_PAIRS}
        documents = addition_explanations(seed=20260925)
        trained = {tuple(p) for row in documents for p in row['pairs']}
        self.assertTrue(held.isdisjoint(trained))

    def test_blending_rejects_wrong_parent_contracts_and_reblending(self):
        from copy import deepcopy
        from scripts.blend_base_probe import validate_lineage
        base = {'stage':'pretrain', 'config':{'vocab_size':100}, 'tokenizer':{'sha256':'tok'}}
        donor = {**deepcopy(base), 'experiment':{'parent_sha256':'parent', 'inference_only':True}}
        validate_lineage(base, donor, 'parent')
        for field, value in [('stage','sft'), ('config',{'vocab_size':101}),
                             ('tokenizer',{'sha256':'other'})]:
            invalid = deepcopy(donor)
            invalid[field] = value
            with self.assertRaises(ValueError):
                validate_lineage(base, invalid, 'parent')
        for field, value in [('parent_sha256','other'), ('inference_only',False),
                             ('blend',{'fraction':.5})]:
            invalid = deepcopy(donor)
            invalid['experiment'][field] = value
            with self.assertRaises(ValueError):
                validate_lineage(base, invalid, 'parent')

    def test_frozen_reasoning_splits_have_correct_oracles_and_no_curriculum_pair_overlap(self):
        from scripts.eval_reasoning_probe import reasoning_cases
        from scripts.probe_base_learning import pair_key
        docs, _, _ = build(count=120, broad_numbers=True, decomposed_addition=True)
        training_pairs = {tuple(pair) for row in docs for pair in row.get('pairs', [])}
        split_pairs = []
        for split in ('dev', 'test'):
            rows = reasoning_cases(split)
            self.assertEqual(len(rows), 64)
            pairs = set()
            for row in rows:
                steps = row['expected_equations']
                a, b = steps[0][0]+steps[1][0], steps[0][1]+steps[1][1]
                pairs.add(pair_key('+', a, b))
                self.assertEqual(a+b, int(row['answer']))
                for left, right, result in steps:
                    self.assertEqual(left+right, result)
                self.assertEqual(steps[2], [steps[0][2], steps[1][2], a+b])
            self.assertEqual(len(pairs), 32)
            self.assertTrue(training_pairs.isdisjoint(pairs))
            split_pairs.append(pairs)
        self.assertTrue(split_pairs[0].isdisjoint(split_pairs[1]))

    def test_reasoned_scoring_requires_correct_steps_and_unambiguous_final_number(self):
        task = {'answer':'76', 'expected_equations':[(50,20,70),(3,3,6),(70,6,76)]}
        steps = 'Tens: 20 + 50 = 70. Units: 3 + 3 = 6. Combine: 6 + 70 = 76. '
        self.assertEqual(score_reasoned_completion(steps+'Answer: 76.',task),
                         {'answer_correct':True,'steps_correct':True})
        wrong = score_reasoned_completion(steps.replace('3 + 3 = 6','3 + 3 = 7')+'Answer: 76.',task)
        self.assertTrue(wrong['answer_correct'])
        self.assertFalse(wrong['steps_correct'])
        for invalid in ('3 + 3 = 6.5', '-3 + 3 = 6', '3 + 3 = 6e2', '3 + 3 = 6,000'):
            self.assertFalse(score_reasoned_completion(
                steps.replace('3 + 3 = 6', invalid)+'Answer: 76.', task)['steps_correct'])
        self.assertFalse(score_reasoned_completion('Answer: 76.',task)['steps_correct'])
        for suffix in ('760.', '76.5', '76x', '-76.', '76.5\nAnswer: 76.'):
            self.assertFalse(score_reasoned_completion(steps+'Answer: '+suffix,task)['answer_correct'])

    def test_addition_audit_catches_wrong_claims_after_a_correct_solution(self):
        from scripts.build_reasoning_curriculum import addition_claim_diagnostics
        task = {'answer': '76', 'expected_equations': [(20, 50, 70), (3, 3, 6), (70, 6, 76)]}
        solution = '20 + 50 = 70. 3 + 3 = 6. 70 + 6 = 76. Answer: 76. '
        for tail in ('18 + 53 = 76.', '18 plus 53 equals 76.',
                     '-23 + 53 = 76.', '23 + 53 = 76.5.', '23 + 53 = 7,600.'):
            with self.subTest(tail=tail):
                output = solution + tail
                self.assertTrue(score_reasoned_completion(output, task)['steps_correct'])
                audit = addition_claim_diagnostics(output)
                self.assertEqual(audit['recognized_addition_claims'], 4)
                self.assertEqual(len(audit['invalid_addition_claims']), 1)
                claim = audit['invalid_addition_claims'][0]
                self.assertEqual(output[claim['start']:claim['end']], claim['claim'])

    def test_addition_audit_accepts_correct_decimals_and_long_integers(self):
        from scripts.build_reasoning_curriculum import addition_claim_diagnostics
        large = '9' * 40
        text = ('0.1 plus 0.2 equals 0.3. -3 + 3 = 0. '
                '1,000 + 2,000 = 3,000. 23 PLUS 53 EQUALS 76. '
                f'{large} + 1 = {int(large) + 1}.')
        audit = addition_claim_diagnostics(text)
        self.assertEqual(audit['recognized_addition_claims'], 5)
        self.assertEqual(audit['invalid_addition_claims'], [])

    def test_broader_number_coverage_keeps_the_exact_evaluation_cases_frozen(self):
        _, _, original = build(count=120)
        docs, _, broadened = build(count=120, broad_numbers=True)
        self.assertEqual(original, broadened)
        held_values = {v for split in broadened.values() for row in split for v in row['values']}
        trained_values = {v for row in docs if 'values' in row for v in row['values']}
        self.assertTrue(held_values.isdisjoint(trained_values))
        self.assertEqual({v//100 for v in trained_values}, set(range(1,10)))

    def test_retrieval_focus_preserves_evaluation_and_training_oracles(self):
        _, _, standard_splits = build(count=120, broad_numbers=True)
        docs, _, focused_splits = build(count=120, broad_numbers=True, retrieval_only=True)
        self.assertEqual(standard_splits, focused_splits)
        context = [row for row in docs if 'template_split' in row]
        self.assertEqual(len(context), 120)
        self.assertEqual({row['kind'] for row in context}, {'retrieve'})
        for row in context:
            self.assertEqual(row['answer'], str(row['values'][row['selected']]))

    def test_place_value_explanations_preserve_answers_and_exclude_test_pairs(self):
        import re
        from scripts.probe_base_learning import build_tasks, pair_key
        from scripts.eval_base_readiness import cases
        _, tasks = build_tasks(worked_examples=True)
        excluded = {tuple(r['pair']) for r in tasks if 'pair' in r}
        excluded.update(pair_key('+', *r['operands']) for split in ('dev','test')
                        for r in cases(split) if 'operands' in r)
        for row in addition_explanations(count=200):
            self.assertEqual(int(row['answer']), row['a'] + row['b'])
            self.assertTrue(excluded.isdisjoint(tuple(p) for p in row['pairs']))
            equations = re.findall(r'(\d+) \+ (\d+) = (\d+)', row['text'])
            self.assertEqual(len(equations), 4)
            for a, b, result in equations:
                self.assertEqual(int(a) + int(b), int(result))

    def test_entities_templates_and_numeric_ranges_are_disjoint(self):
        docs, _, splits = build(count=120)
        train = [x for x in docs if 'template_split' in x]
        for left, right in [(TRAIN_NAMES, DEV_NAMES), (TRAIN_NAMES, TEST_NAMES), (DEV_NAMES, TEST_NAMES)]:
            self.assertFalse(set(left) & set(right))
        numbers = [{n for r in records for n in r['values']} for records in (train, splits['dev'], splits['test'])]
        for a, b in ((0,1),(0,2),(1,2)):
            self.assertFalse(numbers[a] & numbers[b])
        self.assertEqual({r['template_split'] for r in train}, {'train'})
        self.assertFalse({r['prompt'] for r in train} & {r['prompt'] for r in splits['dev']})

    def test_every_grounded_target_has_an_independent_oracle(self):
        docs, _, splits = build(count=300)
        for row in [r for r in docs if 'template_split' in r] + splits['dev'] + splits['test']:
            kind = row.get('kind', row.get('category', '').removeprefix('context_'))
            if kind == 'retrieve':
                expected = str(row['values'][row['selected']])
            else:
                extreme = max(row['values']) if kind == 'maximum' else min(row['values'])
                expected = row['entities'][row['values'].index(extreme)]
            self.assertEqual(row['answer'], expected)
            self.assertEqual(len(set(row['entities'])), 3)
            self.assertEqual(len(set(row['values'])), 3)


if __name__ == '__main__':
    unittest.main()
