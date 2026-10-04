"""First-claim scoring and likelihood choices shared by every BASE probe."""
import unittest

from scripts._base_probe_cases import (
    claim, first_clause, likelihood_choices, numeric_claim, score_completion, suite,
)


def row_with_id(row_id):
    return next(r for r in suite() if r['id'] == row_id)


class FirstClauseTests(unittest.TestCase):
    def test_clause_stops_at_sentence_ends_but_not_decimal_points(self):
        self.assertEqual(first_clause('  43.5 beads. More'), '43.5 beads')
        self.assertEqual(first_clause('Tarin is 431\nQuestion'), 'Tarin is 431')
        self.assertEqual(first_clause('Paris; Lyon'), 'Paris')
        self.assertEqual(first_clause('no terminator'), 'no terminator')


class NumericClaimTests(unittest.TestCase):
    def setUp(self):
        self.row = row_with_id('add_3')  # 38 + 47 = 85

    def test_any_phrasing_of_the_first_claim_counts(self):
        for text in (' 85', ' 85.', ' 85 beads.', ' The answer is 85.', ' It equals 85\nNext',
                     ' 38 + 47 = 85.', ' 85.0', ' 85;'):
            self.assertTrue(score_completion(text, self.row, budget_reached=False), text)

    def test_wrong_hedged_negated_and_later_answers_fail(self):
        for text in (' 86.', ' 850', ' 85.5', ' 85e2', ' -85', ' 85 or 86', ' 85 to 86', ' not 85',
                     ' 85 is not the answer', ' Wrong. 85.', ' 17\n85', ' 12 + 5 = 17. So 85',
                     ' the sum', ''):
            self.assertFalse(score_completion(text, self.row, budget_reached=False), text)

    def test_cut_off_numbers_are_undecided(self):
        self.assertEqual(claim(' 85', self.row, budget_reached=True)['status'], 'undecided')
        self.assertEqual(claim(' The answer is', self.row, budget_reached=True)['status'], 'undecided')
        for text in (' 85.', ' 85 ', ' 85\n', ' 85 beads'):
            self.assertEqual(claim(text, self.row, budget_reached=True)['status'], 'claimed', text)
        self.assertEqual(claim(' 85', self.row, budget_reached=None)['status'], 'undecided')

    def test_correct_first_claim_before_wrong_equation_counts(self):
        for text, value in (('The answer is 85 because 38 + 47 = 84.', '85'),
                            ('85, since 38 + 47 = 84.', '85'),
                            ('85.0 because 38 + 47 = 84.', '85.0')):
            with self.subTest(text=text):
                self.assertEqual(claim(text, self.row, budget_reached=False),
                                 {'status': 'claimed', 'value': value, 'correct': True})

    def test_wrong_first_claim_before_correct_equation_fails(self):
        for text, value in (('The answer is 84 because 38 + 47 = 85.', '84'),
                            ('84, since 38 + 47 = 85.', '84'),
                            ('84.0 because 38 + 47 = 85.', '84.0')):
            with self.subTest(text=text):
                self.assertEqual(claim(text, self.row, budget_reached=False),
                                 {'status': 'claimed', 'value': value, 'correct': False})

    def test_pure_equations_claim_the_first_right_hand_side(self):
        for text, value, correct in (
                ('38 + 47 = 85.', '85', True),
                ('38 + 47 = 84.', '84', False),
                ('(38 + 47) = 85.', '85', True),
                ('38.0 + 47.0 = 85.', '85', True),
                ('38 + 47 = 84 = 85.', '84', False)):
            with self.subTest(text=text):
                self.assertEqual(claim(text, self.row, budget_reached=False),
                                 {'status': 'claimed', 'value': value, 'correct': correct})

    def test_grouped_thousands_and_values(self):
        self.assertEqual(numeric_claim('It is 1,000.', budget_reached=False),
                         {'status': 'claimed', 'value': '1000'})

    def test_grounding_subject_must_be_the_asked_person(self):
        row = {'category': 'grounding', 'answer': '431',
               'prompt': 'The badge code for Tarin is 431. The badge code for Bela is 902.\n'
                         'Question: What is the badge code for Tarin?\nAnswer:'}
        for text in (' 431', ' The badge code for Tarin is 431.', ' Tarin is 431.', ' The badge code is 431.'):
            self.assertTrue(score_completion(text, row, budget_reached=False), text)
        for budget in (False, True, None):
            for text in (' The badge code for Bela is 431.', ' Bela is 431 ', ' the badge code for bela is 902'):
                result = claim(text, row, budget_reached=budget)
                self.assertEqual(result['status'], 'other_subject', (budget, text))
                self.assertFalse(result['correct'])
        self.assertFalse(score_completion(' Tarin is 902. It is 431.', row, budget_reached=False))


class ChoiceClaimTests(unittest.TestCase):
    def setUp(self):
        self.row = row_with_id('fact_france')  # Paris; rivals Lyon, Marseille, Nice

    def test_first_named_answer_in_the_first_clause_counts(self):
        for text in (' Paris.', ' the city of Paris, on the Seine.', ' located in the north: Paris',
                     ' PARIS'):
            self.assertTrue(score_completion(text, self.row, budget_reached=False), text)
        for text in (' Lyon, not Paris.', ' Lyon and Paris.', ' Paris or Lyon.', ' not Paris.',
                     ' a large city. Paris.', ' Parisian.', ''):
            self.assertFalse(score_completion(text, self.row, budget_reached=False), text)

    def test_answer_variants_from_the_accepted_pattern(self):
        hamlet = row_with_id('fact_hamlet')
        for text in (' the English playwright William Shakespeare.', ' Shakespeare'):
            self.assertTrue(score_completion(text, hamlet, budget_reached=False), text)
        self.assertFalse(score_completion(' Christopher Marlowe, not Shakespeare', hamlet, budget_reached=False))
        boil = row_with_id('fact_boil')
        self.assertTrue(score_completion(' 100 °C at sea level.', boil, budget_reached=False))
        self.assertFalse(score_completion(' 0 degrees Celsius.', boil, budget_reached=False))
        water = row_with_id('fact_water')
        self.assertFalse(score_completion(' H2O2.', water, budget_reached=False))

    def test_sequence_continuations_require_the_next_item_first(self):
        alphabet = row_with_id('text_alphabet')
        self.assertTrue(score_completion(' E, F, G', alphabet))
        self.assertFalse(score_completion(' F, E', alphabet))


class LikelihoodChoiceTests(unittest.TestCase):
    def test_every_answer_row_in_the_suite_has_distinct_choices_with_the_answer_first(self):
        for row in suite():
            if row['category'] == 'continuation':
                continue
            choices = likelihood_choices(row)
            self.assertIsNotNone(choices, row['id'])
            self.assertEqual(len(choices), len(set(choices)), row['id'])
            self.assertGreaterEqual(len(choices), 3, row['id'])
            if 'rivals' in row:
                self.assertEqual(choices[0], row['answer'])
                self.assertTrue(score_completion(' ' + row['answer'] + '.', row, budget_reached=False), row['id'])

    def test_numeric_rivals_include_prompt_numbers_and_neighbours(self):
        row = {'category': 'equation', 'answer': '43', 'prompt': '12 + 5 = 17\n31 + 12 ='}
        choices = likelihood_choices(row)
        self.assertEqual(choices[0], '43')
        for rival in ('42', '44', '33', '53', '17', '31', '12', '5'):
            self.assertIn(rival, choices)
        self.assertIsNone(likelihood_choices({'category': 'continuation', 'prompt': 'x'}))


if __name__ == '__main__':
    unittest.main()
