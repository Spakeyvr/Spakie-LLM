import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.download_sft_data as download_sft_data


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
