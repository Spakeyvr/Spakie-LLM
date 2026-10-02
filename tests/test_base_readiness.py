from pathlib import Path
import unittest

from scripts._base_probe_cases import claim, likelihood_choices
from scripts.eval_base_readiness import cases, score_answer

TOKENIZER = Path(__file__).resolve().parents[1] / 'tokenizer' / 'spakie.model'


def first(category, split='dev'):
    return next(r for r in cases(split) if r['category'] == category)


class BaseReadinessTests(unittest.TestCase):
    def test_grounding_judges_the_first_claim_and_its_subject(self):
        row = first('grounding')  # Tarin is 431, Bela is 902
        for text in (' 431', ' The badge code for Tarin is 431.', ' Tarin is 431', ' The badge code is 431.',
                     ' Tarin is 431. Actually it is 902.'):
            self.assertTrue(score_answer(text, row, budget_reached=False), text)
        for text in (' The badge code for Bela is 431.', ' Tarin is 4310.', ' Tarin is 43', ' Tarin is 431.5.',
                     ' Tarin is -431.', ' Tarin is 431e2.', ' Tarin is 431 or 902.', ' Tarin is not 431.',
                     ' Tarin is 902. The answer is 431.', ' Tarin is 431 is not the answer.'):
            self.assertFalse(score_answer(text, row, budget_reached=False), text)

    @unittest.skipUnless(TOKENIZER.exists(), 'trained tokenizer not present in this checkout')
    def test_grounding_budget_fits_a_sentence_form_answer(self):
        from tokenizer.train_tokenizer import SpakieTokenizer
        tok = SpakieTokenizer(str(TOKENIZER))
        for row in (r for r in cases() if r['category'] == 'grounding'):
            name = row['prompt'].rsplit('badge code for ', 1)[1].split('?')[0]
            answer = f" The badge code for {name} is {row['answer']}.\n"
            self.assertLess(len(tok.encode(answer)), row['max_new_tokens'], answer)

    def test_cut_off_answers_are_never_certified(self):
        row = first('grounding')
        for budget in (True, None):
            for text in (' 431', ' 43', ' The badge code for Tarin is 431'):
                self.assertEqual(claim(text, row, budget_reached=budget)['status'], 'undecided', text)
                self.assertFalse(score_answer(text, row, budget_reached=budget), text)
            self.assertTrue(score_answer(' 431\nQuestion: More', row, budget_reached=budget))

    def test_arithmetic_and_facts_accept_any_phrasing_of_a_correct_first_claim(self):
        equation, words = first('equation'), first('word_problem')  # 31 + 12 = 43
        for row in (equation, words):
            for text in (' 43', ' 43.0 beads.', ' 43.\nNext question.', ' The total is 43.', ' 31 + 12 = 43.'):
                self.assertTrue(score_answer(text, row, budget_reached=False), (row['category'], text))
            for text in (' 430', ' 43.5', ' 43,000', ' Wrong. 43', ' -43', ' 17\n31 + 12 = 43', ' 11. 43'):
                self.assertFalse(score_answer(text, row, budget_reached=False), (row['category'], text))
        capital = first('facts')  # Spain: Madrid; rivals Barcelona, Seville, Valencia
        for text in (' Madrid.', ' the city of Madrid.', ' in the city of Madrid, which is', ' Madrid, Spain'):
            self.assertTrue(score_answer(text, capital, budget_reached=False), text)
        for text in (' Barcelona, not Madrid.', ' Barcelona and Madrid.', ' Madrid or Barcelona.', ' Madridista',
                     ' the city of Barcelona. Madrid.', ' in the south of the country'):
            self.assertFalse(score_answer(text, capital, budget_reached=False), text)

    def test_format_instructions_stay_exact(self):
        row = first('format_instruction')
        self.assertEqual(row['scoring'], 'exact')
        self.assertTrue(score_answer(' TARIN\n', row))
        for text in ('TARIN.', 'TARIN\nMore', 'tarin', 'The word TARIN'):
            self.assertFalse(score_answer(text, row))

    def test_every_scored_row_has_likelihood_choices_except_format(self):
        for split in ('dev', 'test'):
            for row in cases(split):
                if 'answer' not in row or row.get('scoring') == 'exact':
                    continue
                choices = likelihood_choices(row)
                self.assertEqual(choices[0], row['answer'])
                self.assertEqual(len(choices), len(set(choices)))
                self.assertGreaterEqual(len(choices), 4, row['prompt'])
        grounding = first('grounding')
        self.assertIn('902', likelihood_choices(grounding))  # the other person's code is a rival

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
