import unittest

from scripts.build_magnitude_curriculum import BANNED, build, deciding_place, verify, worked_pair


class MagnitudeCurriculumTests(unittest.TestCase):
    def test_deciding_place(self):
        self.assertEqual(deciding_place(859, 821), 1)
        self.assertEqual(deciding_place(432, 437), 2)
        self.assertEqual(deciding_place(300, 299), 0)
        with self.assertRaises(ValueError):
            deciding_place(5, 5)

    def test_worked_pair_stops_at_deciding_digit(self):
        text = worked_pair(432, 437)
        self.assertIn('hundreds digits are both 4', text)
        self.assertIn('units digits are 2 and 7; 7 is more than 2', text)
        self.assertTrue(text.endswith('So 437 is greater than 432.'))

    def test_build_is_deterministic_balanced_and_excludes_eval_items(self):
        banned = {frozenset((859, 821))}
        docs = build(900, 7, banned_pairs=banned, banned_names={'Arlo'})
        self.assertEqual(docs, build(900, 7, banned_pairs=banned, banned_names={'Arlo'}))
        kinds = [d['kind'] for d in docs]
        self.assertEqual(kinds.count('answer_first'), kinds.count('worked'))
        places = [d['meta']['deciding_place'] for d in docs if d['kind'] != 'three_way']
        self.assertEqual(places.count('hundreds'), places.count('units'))
        for d in docs:
            self.assertFalse(any(b in d['text'] for b in BANNED))
            self.assertNotIn('Arlo', d['text'])
            if 'pair' in d['meta']:
                self.assertNotEqual(frozenset(d['meta']['pair']), frozenset((859, 821)))

    def test_symmetric_mode_balances_directions_and_keeps_v1_unchanged(self):
        self.assertEqual(build(90, 3), build(90, 3, symmetric=False))
        docs = build(1800, 5, symmetric=True)
        first = [d['text'] for d in docs if d['kind'] == 'answer_first']
        larger = sum(('greater one' in t) or ('bigger' in t) or t.split()[0].isdigit() and 'greater than' in t
                     or 'maximum' in t for t in first)
        smaller = sum(('lesser one' in t) or ('Which is less' in t) or ('less than' in t) for t in first)
        self.assertEqual(larger + smaller, len(first))
        self.assertEqual(larger, smaller)
        worked = [d['text'] for d in docs if d['kind'] == 'worked']
        self.assertTrue(all(', and ' in t and 'is less than' in t for t in worked))
        for d in docs:
            verify(d)

    def test_rich_mode_uses_varied_direction_words_without_eval_template(self):
        docs = build(900, 11, symmetric=True, rich=True)
        first = [d['text'] for d in docs if d['kind'] == 'answer_first']
        self.assertTrue(any('smaller' in t for t in first) and any('lower' in t for t in first))
        self.assertTrue(any('larger' in t for t in first) and any('higher' in t for t in first))
        for d in docs:
            verify(d)
            if d['kind'] == 'answer_first':
                a, b = d['meta']['pair']
                answer = int(d['text'].rstrip('.').split()[-1])
                if any(w in d['text'] for w in ('smaller', 'lesser', 'lower', ' less')):
                    self.assertEqual(answer, min(a, b))
                else:
                    self.assertEqual(answer, max(a, b))

    def test_three_way_only_answers_match_oracle_and_avoid_eval_words(self):
        docs = build(600, 13, three_way_only=True, banned_names={'Arlo'})
        positions = [0, 0, 0]
        for d in docs:
            verify(d)
            m = d['meta']
            self.assertEqual(m['answer'], m['names'][m['values'].index(max(m['values']) if m['want_high'] else min(m['values']))])
            positions[m['names'].index(m['answer'])] += 1
            for word in ('earned', 'marks', 'learner', 'Arlo'):
                self.assertNotIn(word, d['text'])
        self.assertTrue(all(p > 150 for p in positions))
        bad = dict(docs[0], text=docs[0]['text'].rsplit(' ', 1)[0] + ' Nobody.')
        with self.assertRaisesRegex(ValueError, 'Wrong three-way'):
            verify(bad)

    def test_verify_rejects_false_claims_and_leaked_templates(self):
        with self.assertRaisesRegex(ValueError, 'False comparison'):
            verify({'text': '321 is greater than 400.'})
        with self.assertRaisesRegex(ValueError, 'False ordering'):
            verify({'text': 'From largest to smallest the numbers are 300, 500, 100.'})
        with self.assertRaisesRegex(ValueError, 'leaked'):
            verify({'text': 'Compare the numbers 1 and 2.'})


if __name__ == '__main__':
    unittest.main()
