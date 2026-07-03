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
            "DeepSeek-distilled": {"enabled": True, "limit": 0},
            "custom": {"enabled": True, "limit": 0},
            "nemotron_instruction_following_chat_v3": {"enabled": True, "limit": 50000},
            "nemotron_math_v4": {"enabled": True, "limit": 50000},
            "alpaca": {"enabled": False, "limit": 12000},
        }

        sources = download_sft_data.parse_sources(
            "all",
            controlled,
            {"alpaca", "nemotron_instruction_following_chat_v3", "nemotron_math_v4"},
        )

        self.assertEqual(
            sources,
            ["nemotron_instruction_following_chat_v3", "nemotron_math_v4"],
        )

    def test_clean_chat_messages_drops_null_content(self):
        cleaned = download_sft_data.clean_chat_messages(
            [
                {"role": "user", "content": None},
                {"role": "assistant", "content": "Answer."},
            ]
        )

        self.assertEqual(cleaned, [])

    def test_load_no_robots_preserves_clean_chat_messages(self):
        rows = FakeDataset([
            {
                "messages": [
                    {"role": "user", "content": "  Explain photosynthesis.  "},
                    {"role": "assistant", "content": "Plants use light to make sugars."},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "A source-specific persona."},
                    {"role": "user", "content": "Say hello."},
                    {"role": "assistant", "content": "Hello."},
                    {"role": "user", "content": "Say bye."},
                    {"role": "assistant", "content": "Bye."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "skip malformed"},
                    {"role": "assistant", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ]
            },
            {
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
                    "messages": [
                        {"role": "user", "content": "Explain photosynthesis."},
                        {"role": "assistant", "content": "Plants use light to make sugars."},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "Hello."},
                        {"role": "user", "content": "Say bye."},
                        {"role": "assistant", "content": "Bye."},
                    ]
                },
                {
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
            "HuggingFaceTB/smoltalk",
            "all",
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


if __name__ == "__main__":
    unittest.main()
