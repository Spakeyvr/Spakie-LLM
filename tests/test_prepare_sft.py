import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.prepare_sft as prepare_sft


class PrepareSFTTests(unittest.TestCase):
    def test_contains_disallowed_sft_marker_checks_message_content(self):
        for marker in prepare_sft.DISALLOWED_SFT_MARKERS:
            with self.subTest(marker=marker):
                self.assertTrue(
                    prepare_sft.contains_disallowed_sft_marker(
                        [{"role": "assistant", "content": f"before {marker} after"}]
                    )
                )

        self.assertFalse(
            prepare_sft.contains_disallowed_sft_marker(
                [
                    {"role": "user", "content": "Explain function composition."},
                    {"role": "assistant", "content": "It combines outputs and inputs."},
                ]
            )
        )

    def test_load_source_filters_tool_and_reasoning_artifact_examples(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Use this tool."},
                    {"role": "assistant", "content": "<tool_call>\n{\"name\":\"search\"}\n</tool_call>"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Solve this."},
                    {"role": "assistant", "content": "Action: calculate\nObservation: done\nFinal Answer: 4"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are an expert in composing functions."},
                    {"role": "user", "content": "Compose f and g."},
                    {"role": "assistant", "content": "Use f(g(x))."},
                ]
            },
            {"messages": [{"role": "user", "content": "missing assistant"}]},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            with contextlib.redirect_stdout(io.StringIO()) as output:
                examples = prepare_sft.load_source(str(path), "System prompt.", limit=0, seed=42)

        self.assertEqual(
            examples,
            [
                {
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                }
            ],
        )
        log = output.getvalue()
        self.assertIn("filtered 3 tool/template artifact examples", log)
        self.assertIn("skipped 1 malformed lines", log)

    def test_build_assistant_seed_examples_repeats_and_injects_system(self):
        examples = prepare_sft.build_assistant_seed_examples("System prompt.", repeats=2)

        self.assertEqual(len(examples), len(prepare_sft.ASSISTANT_BEHAVIOR_SEEDS) * 2)
        self.assertEqual(examples[0]["messages"][0], {"role": "system", "content": "System prompt."})
        self.assertEqual(examples[0]["messages"][1], {"role": "user", "content": "Hi"})
        self.assertEqual(
            examples[0]["messages"][2],
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        )

    def test_build_assistant_seed_examples_can_omit_system(self):
        examples = prepare_sft.build_assistant_seed_examples(None, repeats=1)

        self.assertEqual(examples[0]["messages"][0], {"role": "user", "content": "Hi"})
        self.assertNotIn("system", {msg["role"] for msg in examples[0]["messages"]})
        self.assertEqual(prepare_sft.build_assistant_seed_examples("System prompt.", repeats=0), [])

    def test_build_pair_seed_examples_repeats_and_injects_system(self):
        examples = prepare_sft.build_pair_seed_examples(
            (("What's Python", "Python is a programming language."),),
            "System prompt.",
            repeats=2,
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["messages"][0], {"role": "system", "content": "System prompt."})
        self.assertEqual(examples[0]["messages"][1], {"role": "user", "content": "What's Python"})
        self.assertEqual(
            examples[0]["messages"][2],
            {"role": "assistant", "content": "Python is a programming language."},
        )

    def test_build_pair_seed_examples_can_omit_system(self):
        examples = prepare_sft.build_pair_seed_examples(
            (("Explain sleep", "Sleep helps the body rest."),),
            None,
            repeats=1,
        )

        self.assertEqual(examples[0]["messages"][0], {"role": "user", "content": "Explain sleep"})
        self.assertNotIn("system", {msg["role"] for msg in examples[0]["messages"]})
        self.assertEqual(prepare_sft.build_pair_seed_examples(prepare_sft.ANTI_ECHO_SEEDS, None, repeats=0), [])


if __name__ == "__main__":
    unittest.main()
