import unittest

from scripts.build_complete_multitask_v3 import build_rows, eval_prompts, normalize


class CompleteMultitaskV3Tests(unittest.TestCase):
    def test_rows_are_substantial_unique_and_valid(self):
        rows = build_rows()
        self.assertEqual(len(rows), 5400)
        prompts = [normalize(item["messages"][0]["content"]) for item in rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        for item in rows:
            self.assertEqual([m["role"] for m in item["messages"]], ["user", "assistant"])
            self.assertTrue(item["messages"][1]["content"].strip())

    def test_no_exact_eval_prompt_overlap(self):
        prompts = {normalize(item["messages"][0]["content"]) for item in build_rows()}
        self.assertFalse(prompts & eval_prompts())

    def test_held_out_capital_entities_are_absent(self):
        prompts = "\n".join(item["messages"][0]["content"] for item in build_rows())
        for country in ("France", "Peru", "Iceland", "Vietnam", "Morocco", "Finland", "Chile", "Croatia"):
            self.assertNotIn(country, prompts)


if __name__ == "__main__":
    unittest.main()
