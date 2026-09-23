import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_sft_seed_data


class BuildSFTSeedDataTests(unittest.TestCase):
    def test_main_writes_size_independent_identity_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(build_sft_seed_data, "SpakieConfig", return_value=SimpleNamespace(chat_raw_dir=directory)):
                self.assertEqual(build_sft_seed_data.main(), 0)
            files = {file.name for file in Path(directory).glob("*.jsonl")}
            self.assertEqual(files, {"spakie_identity.jsonl", "assistant_behavior.jsonl", "anti_echo.jsonl", "factual_repairs.jsonl"})
            rows = [json.loads(line) for line in (Path(directory) / "spakie_identity.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all(row["source"] == "spakie_identity" for row in rows))
            for row in rows:
                self.assertNotRegex(row["messages"][-1]["content"], r"\d+[- ]million|Spakie-\d+")

    def test_targeted_builder_also_uses_size_independent_identity(self):
        from scripts.build_targeted_data import build_sft
        rows = build_sft([], seed=7)
        identities = []
        for row in rows:
            for message in row["messages"]:
                if message["role"] == "assistant":
                    self.assertNotRegex(message["content"], r"Spakie-\d+|180[- ]million")
            if any(message["content"] == "Who are you?" for message in row["messages"]):
                identities.append(row["messages"][-1]["content"])
        self.assertTrue(identities)
        self.assertTrue(all("Spakie" in answer for answer in identities))

    def test_write_source_labels_and_deduplicates_rows(self):
        example = {
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Spakie."},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "identity.jsonl"
            count = build_sft_seed_data.write_source(
                str(path), "spakie_identity", [example, example]
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(rows, [{"source": "spakie_identity", **example}])


if __name__ == "__main__":
    unittest.main()
