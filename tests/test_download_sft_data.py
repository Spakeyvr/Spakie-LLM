import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.download_sft_data as download_sft_data
from configs.default import SpakieConfig


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, seed: int, **_kwargs):
        return self

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])

    def take(self, limit: int):
        return FakeDataset(self.rows[:limit])


class DownloadSFTDataTests(unittest.TestCase):
    def test_download_limit_can_exceed_merge_limit(self):
        config = SpakieConfig()
        config.sft_source_limits = {
            "smoltalk": {"enabled": True, "limit": 40000, "download_limit": 150000}
        }
        self.assertEqual(config.sft_source_limit("smoltalk"), 40000)
        self.assertEqual(config.sft_source_download_limit("smoltalk"), 150000)

    def test_main_skips_zero_limit_sources(self):
        # Use a controlled limits config (decoupled from the live YAML, whose
        # values legitimately change): squad disabled, triviaqa enabled.
        controlled = SpakieConfig()
        controlled.sft_source_limits = {"squad": 0, "triviaqa": 3000}
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "download_sft_data.py",
                "--sources",
                "squad,triviaqa",
                "--output-dir",
                tmpdir,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_sft_data, "SpakieConfig", return_value=controlled),
                patch.object(download_sft_data, "load_squad") as load_squad,
                patch.object(download_sft_data, "load_triviaqa", return_value=[]) as load_triviaqa,
            ):
                download_sft_data.main()

        load_squad.assert_not_called()
        load_triviaqa.assert_called_once()

    def test_parse_sources_all_uses_enabled_downloadable_sources(self):
        controlled = SpakieConfig()
        controlled.sft_source_limits = {
            "DeepSeek-distill-V2": {"enabled": True, "limit": 0},
            "custom": {"enabled": True, "limit": 0},
            "nemotron_instruction_following_chat_v3": {"enabled": True, "limit": 50000},
            "alpaca": {"enabled": False, "limit": 12000},
        }

        sources = download_sft_data.parse_sources(
            "all",
            controlled,
            {"alpaca", "nemotron_instruction_following_chat_v3"},
        )

        self.assertEqual(
            sources,
            ["nemotron_instruction_following_chat_v3"],
        )

    def test_clean_chat_messages_drops_null_content(self):
        cleaned = download_sft_data.clean_chat_messages(
            [
                {"role": "user", "content": None},
                {"role": "assistant", "content": "Answer."},
            ]
        )

        self.assertEqual(cleaned, [])

    def test_clean_chat_messages_preserves_code_line_structure(self):
        cleaned = download_sft_data.clean_chat_messages(
            [
                {"role": "user", "content": "Write Python."},
                {
                    "role": "assistant",
                    "content": "```python\ndef add(a, b):\n    return a + b\n```",
                },
            ]
        )
        self.assertEqual(
            cleaned[-1]["content"],
            "```python\ndef add(a, b):\n    return a + b\n```",
        )

    def test_load_no_robots_preserves_clean_chat_messages(self):
        rows = FakeDataset([
            {
                "category": "Open QA",
                "messages": [
                    {"role": "user", "content": "  Explain photosynthesis.  "},
                    {"role": "assistant", "content": "Plants use light to make sugars."},
                ]
            },
            {
                "category": "Chat",
                "messages": [
                    {"role": "system", "content": "A source-specific persona."},
                    {"role": "user", "content": "Say hello."},
                    {"role": "assistant", "content": "Hello."},
                    {"role": "user", "content": "Say bye."},
                    {"role": "assistant", "content": "Bye."},
                ]
            },
            {
                "category": "Closed QA",
                "messages": [
                    {"role": "user", "content": "skip malformed"},
                    {"role": "assistant", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ]
            },
            {
                "category": "Closed QA",
                "messages": [
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
        ])

        with patch.object(download_sft_data, "load_dataset", return_value=rows) as load_dataset:
            examples = download_sft_data.load_no_robots(limit=10, seed=42)

        load_dataset.assert_called_once_with("HuggingFaceH4/no_robots", split="train")
        self.assertEqual(
            examples,
            [
                {
                    "category": "Open QA",
                    "messages": [
                        {"role": "user", "content": "Explain photosynthesis."},
                        {"role": "assistant", "content": "Plants use light to make sugars."},
                    ]
                },
                {
                    "category": "Closed QA",
                    "messages": [
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                },
            ],
        )

    def test_load_smoltalk_streams_and_preserves_task_system_instruction(self):
        rows = FakeDataset([
            {
                "messages": [
                    {"role": "system", "content": "Rewrite the input to be concise."},
                    {"role": "user", "content": "This is much too long."},
                    {"role": "assistant", "content": "Too long."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Name one color."},
                    {"role": "assistant", "content": "Blue."},
                ]
            },
        ])

        with patch.object(download_sft_data, "load_dataset", return_value=rows) as load_dataset:
            examples = download_sft_data.load_smoltalk(limit=2, seed=7)

        load_dataset.assert_called_once_with(
            "HuggingFaceTB/smol-smoltalk",
            split="train",
            streaming=True,
        )
        self.assertEqual(
            examples,
            [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Rewrite the input to be concise.\n\nThis is much too long.",
                        },
                        {"role": "assistant", "content": "Too long."},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Name one color."},
                        {"role": "assistant", "content": "Blue."},
                    ]
                },
            ],
        )

    def test_source_limit_counts_usable_examples_not_raw_rows(self):
        rows = FakeDataset(
            [
                {"messages": "malformed"},
                {
                    "messages": [
                        {"role": "user", "content": "First usable."},
                        {"role": "assistant", "content": "One."},
                    ]
                },
                {"messages": [{"role": "user", "content": "incomplete"}]},
                {
                    "messages": [
                        {"role": "user", "content": "Second usable."},
                        {"role": "assistant", "content": "Two."},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Must not be consumed."},
                        {"role": "assistant", "content": "Three."},
                    ]
                },
            ]
        )

        with patch.object(download_sft_data, "load_dataset", return_value=rows):
            examples = download_sft_data.load_no_robots(limit=2, seed=42)

        self.assertEqual(len(examples), 2)
        self.assertEqual(
            [example["messages"][0]["content"] for example in examples],
            ["First usable.", "Second usable."],
        )

    def test_main_returns_failure_when_any_source_fails(self):
        controlled = SpakieConfig()
        controlled.sft_source_limits = {"squad": 2, "triviaqa": 2}
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "download_sft_data.py",
                "--sources",
                "squad,triviaqa",
                "--output-dir",
                tmpdir,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_sft_data, "SpakieConfig", return_value=controlled),
                patch.object(download_sft_data, "load_squad", side_effect=RuntimeError("offline")),
                patch.object(download_sft_data, "load_triviaqa", return_value=[]),
            ):
                result = download_sft_data.main()

        self.assertEqual(result, 1)

    def test_only_nemotron_chat_v3_marks_final_assistant_as_training_target(self):
        rows = FakeDataset([
            {
                "messages": [
                    {"role": "user", "content": "First question."},
                    {"role": "assistant", "content": "Context response."},
                    {"role": "user", "content": "Follow-up question."},
                    {"role": "assistant", "content": "Intended target."},
                ]
            }
        ])

        with patch.object(download_sft_data, "load_dataset", return_value=rows):
            chat_examples = download_sft_data.load_nemotron_instruction_following_chat_v3(
                limit=10,
                seed=42,
            )

        self.assertEqual(
            chat_examples[0]["messages"],
            [
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": "Context response.", "train": False},
                {"role": "user", "content": "Follow-up question."},
                {"role": "assistant", "content": "Intended target.", "train": True},
            ],
        )


if __name__ == "__main__":
    unittest.main()
