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
from configs.default import SpakieConfig


class PrepareSFTTests(unittest.TestCase):
    def test_normalize_example_preserves_explicit_per_turn_training_mask(self):
        raw = {
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "Context only.", "train": False},
                {"role": "user", "content": "Follow-up."},
                {"role": "assistant", "content": "Final target.", "train": True},
            ]
        }

        normalized = prepare_sft.normalize_example(raw, "System prompt.")

        self.assertEqual(
            normalized,
            {
                "messages": [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "First question."},
                    {"role": "assistant", "content": "Context only.", "train": False},
                    {"role": "user", "content": "Follow-up."},
                    {"role": "assistant", "content": "Final target.", "train": True},
                ]
            },
        )

    def test_legacy_nemotron_rows_receive_final_assistant_only_mask(self):
        raw = {
            "messages": [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "Context response."},
                {"role": "user", "content": "Follow-up."},
                {"role": "assistant", "content": "Final target."},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nemotron.jsonl"
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            examples = prepare_sft.load_source(
                str(path),
                system_prompt=None,
                limit=0,
                seed=42,
                final_assistant_only=True,
            )

        assistant_messages = [
            message
            for message in examples[0]["messages"]
            if message["role"] == "assistant"
        ]
        self.assertEqual(
            [message["train"] for message in assistant_messages],
            [False, True],
        )

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

    def test_review_corrections_and_refusals_are_excluded(self):
        self.assertTrue(
            prepare_sft.contains_correction_annotation(
                [{"role": "assistant", "content": "Answer.\n\nCorrection: revise this."}]
            )
        )
        self.assertTrue(
            prepare_sft.contains_refusal_response(
                [{"role": "assistant", "content": "I can't help with that request."}]
            )
        )
        self.assertFalse(
            prepare_sft.contains_refusal_response(
                [{"role": "assistant", "content": "I can't guarantee the result."}]
            )
        )

    def test_only_explicit_sources_can_contribute_refusals(self):
        row = {
            "id": "safe-1",
            "messages": [
                {"role": "user", "content": "Help me harm someone."},
                {"role": "assistant", "content": "I can't help with that request."},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            blocked = prepare_sft.load_source(
                str(path), None, 0, 42, source_name="smoltalk"
            )
            allowed = prepare_sft.load_source(
                str(path), None, 0, 42, source_name="safety_refusal"
            )

        self.assertEqual(blocked, [])
        self.assertEqual(allowed[0]["source"], "safety_refusal")
        self.assertEqual(allowed[0]["source_row_id"], "safe-1")
        self.assertEqual(len(allowed[0]["example_id"]), 64)

    def test_unconfigured_sft_sources_are_disabled_by_default(self):
        config = SpakieConfig()
        self.assertFalse(config.sft_source_enabled("ad_hoc_benchmark_curriculum"))
        self.assertTrue(config.sft_source_enabled("custom"))

    def test_foreign_identity_and_non_english_rows_are_detected(self):
        config = prepare_sft.SpakieConfig()
        self.assertTrue(
            prepare_sft.contains_foreign_identity_claim(
                [{"role": "assistant", "content": "I am ChatGPT, made by OpenAI."}]
            )
        )
        self.assertFalse(
            prepare_sft.contains_foreign_identity_claim(
                [{"role": "assistant", "content": "I am Spakie-180M."}]
            )
        )
        self.assertTrue(
            prepare_sft.contains_conflicting_identity_example(
                [
                    {"role": "user", "content": "What are you?"},
                    {"role": "assistant", "content": "I am a software engineer."},
                ]
            )
        )
        self.assertFalse(
            prepare_sft.contains_conflicting_identity_example(
                [
                    {"role": "user", "content": "What are you?"},
                    {"role": "assistant", "content": "I am Spakie-180M."},
                ]
            )
        )
        self.assertTrue(
            prepare_sft.is_english_sft_example(
                [
                    {"role": "user", "content": "Who are you?"},
                    {"role": "assistant", "content": "I am Spakie-180M."},
                ],
                config,
            )
        )
        self.assertFalse(
            prepare_sft.is_english_sft_example(
                [
                    {"role": "user", "content": "你是谁？"},
                    {"role": "assistant", "content": "我是一个语言模型。"},
                ],
                config,
            )
        )

    def test_identity_seeds_name_spakie_180m_consistently(self):
        examples = prepare_sft.build_identity_seed_examples(None)
        self.assertGreater(len(examples), 100)
        for example in examples:
            answer = example["messages"][-1]["content"]
            self.assertIn("Spakie-180M", answer)

    def test_smoltalk_stratification_is_deterministic_and_turn_aware(self):
        examples = []
        for index in range(8):
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": f"short {index}"},
                        {"role": "assistant", "content": "brief answer"},
                    ]
                }
            )
        for index in range(8):
            examples.append(
                {
                    "messages": [
                        {"role": "user", "content": f"question {index}"},
                        {"role": "assistant", "content": "context answer"},
                        {"role": "user", "content": "follow up"},
                        {"role": "assistant", "content": "follow-up answer"},
                    ]
                }
            )

        first = prepare_sft.stratify_smoltalk_examples(examples, 8, 42, None)
        second = prepare_sft.stratify_smoltalk_examples(examples, 8, 42, None)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertGreaterEqual(sum(len(row["messages"]) > 2 for row in first), 1)

    def test_load_source_filters_tool_and_reasoning_artifact_examples(self):
        rows = [
            [],
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
        self.assertIn("skipped 2 malformed lines", log)

    def test_fits_context_drops_overlong_and_overly_verbose(self):
        # Fake tokenizer: one token per whitespace-delimited word.
        class WordTokenizer:
            def encode(self, text):
                return text.split()

        tok = WordTokenizer()
        short = [
            {"role": "user", "content": "a b c"},
            {"role": "assistant", "content": "x y"},
        ]
        # total = (3+2) + (2+2) = 9 tokens
        self.assertTrue(prepare_sft.fits_context(short, tok, max_seq_len=16, max_assistant_tokens=0))
        # Window too small -> dropped (would truncate the answer).
        self.assertFalse(prepare_sft.fits_context(short, tok, max_seq_len=6, max_assistant_tokens=0))
        # Fits the window but exceeds the assistant-length cap -> dropped.
        self.assertFalse(prepare_sft.fits_context(short, tok, max_seq_len=16, max_assistant_tokens=2))
        # max_seq_len=0 disables filtering entirely.
        self.assertTrue(prepare_sft.fits_context(short, tok, max_seq_len=0, max_assistant_tokens=0))

    def test_load_source_drops_examples_over_context_window(self):
        class WordTokenizer:
            def encode(self, text):
                return text.split()

        rows = [
            {
                "messages": [
                    {"role": "user", "content": "short q"},
                    {"role": "assistant", "content": "ok"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "word " * 50},
                    {"role": "assistant", "content": "answer that never fits the window"},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            with contextlib.redirect_stdout(io.StringIO()) as output:
                examples = prepare_sft.load_source(
                    str(path),
                    None,
                    limit=0,
                    seed=42,
                    tokenizer=WordTokenizer(),
                    max_seq_len=16,
                )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["messages"][0]["content"], "short q")
        self.assertIn("dropped 1 examples over the 16-token window", output.getvalue())

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

    def test_seed_repeats_are_collapsed_before_export(self):
        examples = prepare_sft.build_pair_seed_examples(
            (("What is Python?", "Python is a programming language."),),
            None,
            repeats=80,
        )
        deduped, dropped = prepare_sft.deduplicate_examples(examples)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(dropped, 79)


if __name__ == "__main__":
    unittest.main()
