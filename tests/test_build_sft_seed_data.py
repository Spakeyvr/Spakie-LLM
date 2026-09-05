import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_sft_seed_data


class BuildSFTSeedDataTests(unittest.TestCase):
    def test_write_source_labels_and_deduplicates_rows(self):
        example = {
            "messages": [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am Spakie-180M."},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "identity.jsonl"
            count = build_sft_seed_data.write_source(
                str(path), "spakie_180m_identity", [example, example]
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(rows, [{"source": "spakie_180m_identity", **example}])


if __name__ == "__main__":
    unittest.main()
