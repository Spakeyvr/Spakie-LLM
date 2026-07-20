import unittest

from scripts.build_sft_experiment_mix import (
    build_mix,
    canonical_user_prompt,
    conversation_signature,
)


def row(prompt, answer):
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}


class BuildSFTExperimentMixTests(unittest.TestCase):
    def test_target_repeats_and_leakage_filter(self):
        baseline = [row("Broad prompt", "Broad answer")]
        targets = [("target.jsonl", [row("Skill prompt", "Skill answer"), row("Eval prompt", "No")], 3)]
        mixed, report = build_mix(
            baseline,
            targets,
            baseline_sample=0,
            seed=7,
            blocked_prompts={"eval prompt"},
        )
        self.assertEqual(len(mixed), 4)
        self.assertEqual(report["evaluation_leakage_rows_dropped"], 1)
        self.assertEqual(report["targets"][0]["weighted_rows"], 3)

    def test_target_deduplication_uses_conversation_content(self):
        duplicate_a = row("Same", "Answer")
        duplicate_b = row(" same ", " answer ")
        mixed, report = build_mix(
            [],
            [("a.jsonl", [duplicate_a, duplicate_b], 1)],
            baseline_sample=0,
            seed=1,
            blocked_prompts=set(),
        )
        self.assertEqual(len(mixed), 1)
        self.assertEqual(report["targets"][0]["duplicates_with_other_targets"], 1)
        self.assertEqual(conversation_signature(duplicate_a), conversation_signature(duplicate_b))

    def test_leakage_filter_removes_generic_prompt_wrappers(self):
        wrapped = row(
            "Please answer carefully. Return only valid JSON with keys city and country for Kyoto in Japan.",
            '{"city":"Kyoto","country":"Japan"}',
        )
        mixed, report = build_mix(
            [],
            [("target.jsonl", [wrapped], 1)],
            baseline_sample=0,
            seed=1,
            blocked_prompts={
                canonical_user_prompt(
                    "Return only valid JSON with keys city and country for Kyoto in Japan."
                )
            },
        )
        self.assertEqual(mixed, [])
        self.assertEqual(report["evaluation_leakage_rows_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
