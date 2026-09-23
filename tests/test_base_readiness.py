import unittest

from scripts.eval_base_readiness import cases, score_answer, grounding_diagnostic


class BaseReadinessTests(unittest.TestCase):
    def test_grounding_diagnostic_separates_answer_from_format_and_trailing_claims(self):
        row = next(r for r in cases() if r['category'] == 'grounding')
        for text in ('431', 'The badge code for Tarin is 431.', 'Tarin is 431',
                     'The badge code is 431.'):
            result = grounding_diagnostic(text, row, generation_budget_reached=False)
            self.assertTrue(result['whole_response_supported'], text)
        self.assertFalse(score_answer('The badge code for Tarin is 431.', row))
        for text in ('The badge code for Bela is 431.', 'Tarin is 4310.',
                     'Tarin is 43', 'Tarin is 431.5.', 'Tarin is -431.',
                     'Tarin is 431e2.', 'Tarin is 431 or 902.',
                     'Tarin is not 431.', 'Tarin is 902. The answer is 431.',
                     'Tarin is 431 is not the answer.'):
            self.assertFalse(grounding_diagnostic(text, row, generation_budget_reached=False)
                             ['first_assertion_correct'], text)
        result = grounding_diagnostic('Tarin is 431. Actually it is 902.', row)
        self.assertTrue(result['first_assertion_correct'])
        self.assertFalse(result['whole_response_supported'])
        self.assertEqual(result['unassessed_remainder'], 'Actually it is 902.')

    def test_grounding_budget_exhaustion_and_unknown_endings_are_not_certified(self):
        row = next(r for r in cases() if r['category'] == 'grounding')
        for budget in (True, None):
            for text in ('431', '43', 'The badge code for Tarin is 431'):
                result = grounding_diagnostic(text, row, generation_budget_reached=budget)
                self.assertIsNone(result['first_assertion_correct'])
                self.assertFalse(result['whole_response_supported'])
            result = grounding_diagnostic('431\nQuestion: More', row,
                                          generation_budget_reached=budget)
            self.assertTrue(result['first_assertion_correct'])
            self.assertFalse(result['whole_response_supported'])
        self.assertFalse(grounding_diagnostic('43', row, generation_budget_reached=False)
                         ['first_assertion_correct'])

    def test_scores_reject_embedded_answers_wrong_numbers_and_extra_format_text(self):
        numeric = {'answer': '43', 'scoring': 'number'}
        for text in ('430', '43.5', '43.5x', '43e2', '43,000', 'Wrong. 43', '-43'):
            self.assertFalse(score_answer(text, numeric), text)
        self.assertTrue(score_answer(' 43.0 beads.', numeric))
        self.assertTrue(score_answer(' 43.\nNext question.', numeric))
        self.assertTrue(score_answer(' 1,000.', {'answer':'1000','scoring':'number'}))
        fact = {'answer': 'Madrid', 'scoring': 'prefix'}
        self.assertTrue(score_answer(' Madrid.', fact))
        self.assertFalse(score_answer('Barcelona, not Madrid.', fact))
        self.assertFalse(score_answer('Madridista', fact))
        capital={**fact,'category':'facts','prompt':'The capital of Spain is'}
        self.assertTrue(score_answer('the city of Madrid.',capital))
        self.assertFalse(score_answer('the city of Barcelona. Madrid.',capital))
        exact = {'answer': 'TARIN', 'scoring': 'exact'}
        self.assertTrue(score_answer(' TARIN\n', exact))
        for text in ('TARIN.', 'TARIN\nMore', 'tarin'):
            self.assertFalse(score_answer(text, exact))

    def test_splits_are_disjoint_and_arithmetic_oracles_are_correct(self):
        dev, test = cases('dev'), cases('test')
        self.assertFalse({r.get('prompt', r.get('good')) for r in dev} &
                         {r.get('prompt', r.get('good')) for r in test})
        for rows in (dev, test):
            self.assertEqual(len(rows), len({r['id'] for r in rows}))
            by_pair = {}
            for row in rows:
                if 'operands' in row:
                    self.assertEqual(int(row['answer']), sum(row['operands']))
                    by_pair.setdefault(tuple(row['operands']), set()).add(row['category'])
            self.assertTrue(all(v == {'equation','word_problem'} for v in by_pair.values()))


if __name__ == '__main__':
    unittest.main()
