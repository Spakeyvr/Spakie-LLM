import unittest

from scripts.build_instruction_reasoning_v2 import (
    benchmark_rows,
    build_compound_instruction_rows,
    build_grounded_multitask_rows,
    build_math_reasoning_rows,
    build_structured_coding_rows,
    normalized,
)


class InstructionReasoningV2Tests(unittest.TestCase):
    def test_benchmark_sizes_and_unique_ids(self):
        core = benchmark_rows("core")
        fresh = benchmark_rows("fresh")
        self.assertEqual(len(core), 96)
        self.assertEqual(len(fresh), 48)
        ids = [row["id"] for row in core + fresh]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(row.get("unit_checks") for row in core + fresh))

    def test_training_families_are_substantial_and_valid(self):
        families = [
            build_compound_instruction_rows(),
            build_math_reasoning_rows(),
            build_structured_coding_rows(),
            build_grounded_multitask_rows(),
        ]
        self.assertGreater(sum(map(len, families)), 7000)
        for rows in families:
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual([m["role"] for m in row["messages"]], ["user", "assistant"])
                self.assertTrue(row["messages"][0]["content"].strip())
                self.assertTrue(row["messages"][1]["content"].strip())

    def test_no_exact_prompt_leakage_between_generated_train_and_eval(self):
        train_prompts = {
            normalized(row["messages"][0]["content"])
            for rows in (
                build_compound_instruction_rows(),
                build_math_reasoning_rows(),
                build_structured_coding_rows(),
                build_grounded_multitask_rows(),
            )
            for row in rows
        }
        for row in benchmark_rows("core") + benchmark_rows("fresh"):
            self.assertNotIn(normalized(row["prompt"]), train_prompts)

    def test_capital_of_france_compound_prompt_is_eval_only(self):
        core = benchmark_rows("core")
        prompt = "What’s the capital of France and why’s it so special?"
        self.assertEqual(core[0]["prompt"], prompt)
        training_text = "\n".join(
            row["messages"][0]["content"] for row in build_compound_instruction_rows()
        )
        self.assertNotIn(prompt, training_text)


if __name__ == "__main__":
    unittest.main()
