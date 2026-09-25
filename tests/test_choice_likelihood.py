import unittest

from scripts.eval_choice_likelihood import choices_for, continuation_ids, summarize


class FakeTokenizer:
    def __init__(self, merge_boundary=False):
        self.merge_boundary = merge_boundary

    def encode(self, text):
        ids = [ord(c) for c in text]
        if self.merge_boundary and text.endswith(' Bo'):
            ids[-4:-2] = [999]  # the prompt's last piece changes once the choice is appended
        return ids


def record(category, pair_id, correct, chosen, answer, n=3, margin=1.):
    return {'category': category, 'pair_id': pair_id, 'choice_correct': correct, 'chosen_index': chosen,
            'answer_index': answer, 'choice_logprobs': [0.] * n, 'answer_margin': margin}


class ChoiceLikelihoodTests(unittest.TestCase):
    def test_choices_use_entities_or_values_matching_the_answer(self):
        self.assertEqual(choices_for({'id': 'a', 'answer': 'Gavi', 'entities': ['Haro', 'Gavi'], 'values': [1, 2]}),
                         ['Haro', 'Gavi'])
        self.assertEqual(choices_for({'id': 'b', 'answer': '889', 'entities': ['Dori'], 'values': [862, 889]}),
                         ['862', '889'])
        with self.assertRaises(ValueError):
            choices_for({'id': 'c', 'answer': 'Nobody', 'entities': ['Haro'], 'values': [1]})

    def test_continuation_requires_stable_prompt_prefix(self):
        prompt, target = continuation_ids(FakeTokenizer(), 'is', 'Bo')
        self.assertEqual(prompt, [ord('i'), ord('s')])
        self.assertEqual(target, [ord(' '), ord('B'), ord('o')])
        with self.assertRaisesRegex(ValueError, 'Unstable'):
            continuation_ids(FakeTokenizer(merge_boundary=True), 'is', 'Bo')

    def test_summary_counts_pairs_chance_and_positions(self):
        rows = [record('max', 0, True, 0, 0), record('max', 0, False, 0, 2, margin=-2.),
                record('max', 1, True, 1, 1), record('max', 1, True, 2, 2, margin=3.)]
        out = summarize(rows)['max']
        self.assertEqual((out['correct'], out['total'], out['complete_pairs'], out['pairs']), (3, 4, 1, 2))
        self.assertAlmostEqual(out['chance_accuracy'], 1 / 3)
        self.assertAlmostEqual(out['mean_answer_margin'], 0.75)
        self.assertEqual(out['chosen_position'], {'0': 2, '1': 1, '2': 1})


if __name__ == '__main__':
    unittest.main()
