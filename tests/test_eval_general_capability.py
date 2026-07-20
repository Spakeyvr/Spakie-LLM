import unittest

from scripts.eval_general_capability import contains_term, score_answer, summarize_results


class GeneralCapabilityScoringTests(unittest.TestCase):
    def test_all_of_requires_every_subcheck(self):
        row = {
            "id": "composite",
            "check": {
                "type": "all_of",
                "value": [
                    {"type": "contains_all", "value": ["Paris"]},
                    {"type": "contains_any", "value": ["culture", "history"]},
                ],
            },
        }
        self.assertTrue(score_answer(row, "Paris is known for its culture.")[0])
        self.assertFalse(score_answer(row, "Paris is the capital.")[0])

    def test_term_matching_uses_boundaries(self):
        self.assertTrue(contains_term("The clothes dry outside.", "dry"))
        self.assertFalse(contains_term("Put them in the dryer.", "dry"))
        self.assertFalse(contains_term("The answer is 25.", "5"))

    def test_reject_any_overrides_required_terms(self):
        row = {
            "id": "brakes",
            "category": "science",
            "check": {
                "type": "contains_all",
                "value": ["friction"],
                "reject_any": ["reduce friction"],
            },
        }
        self.assertEqual(score_answer(row, "Brakes reduce friction.")[0], False)

    def test_choice_requires_leading_choice_and_answer_text(self):
        row = {
            "id": "choice",
            "category": "science",
            "check": {"type": "choice", "letter": "B", "answer_terms": ["friction"]},
        }
        self.assertEqual(score_answer(row, "A. Gravity; B. friction")[0], False)
        self.assertEqual(score_answer(row, "B. They create friction.")[0], True)

    def test_json_can_check_expected_values(self):
        row = {
            "id": "json",
            "category": "format",
            "check": {
                "type": "json",
                "required_keys": ["name", "age"],
                "expected": {"name": "Amina", "age": 24},
            },
        }
        self.assertEqual(score_answer(row, '{"name":"Amina","age":25}')[0], False)
        self.assertEqual(score_answer(row, '{"name":"Amina","age":24}')[0], True)

    def test_exact_json_checks_root_value(self):
        row = {
            "id": "json-array",
            "category": "format",
            "check": {"type": "exact_json", "value": [2, 4, 6]},
        }
        self.assertEqual(score_answer(row, "[2, 4, 6]")[0], True)
        self.assertEqual(score_answer(row, "[2, 6, 4]")[0], False)

    def test_macro_accuracy_balances_categories(self):
        rows = [
            {"id": "a1", "category": "a", "answer": "yes", "check": {"type": "exact", "value": "yes"}},
            {"id": "a2", "category": "a", "answer": "yes", "check": {"type": "exact", "value": "yes"}},
            {"id": "b1", "category": "b", "answer": "no", "check": {"type": "exact", "value": "yes"}},
        ]
        summary = summarize_results(rows)
        self.assertAlmostEqual(summary["micro_accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["macro_accuracy"], 0.5)

    def test_manual_rating_is_counted_when_present(self):
        rows = [{
            "id": "manual",
            "category": "conversation",
            "answer": "Nice work!",
            "check": {"type": "manual"},
            "manual_passed": True,
        }]
        summary = summarize_results(rows)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["manual"], 0)


if __name__ == "__main__":
    unittest.main()
