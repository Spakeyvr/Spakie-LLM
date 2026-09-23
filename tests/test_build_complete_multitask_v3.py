import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_complete_multitask_v3 as builder
from scripts.build_complete_multitask_v3 import build_rows, eval_prompts, normalize
from scripts.build_instruction_reasoning_v2 import benchmark_rows


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
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            expected = set()
            for split in ("core", "fresh"):
                rows = benchmark_rows(split)
                expected.update(normalize(row["prompt"]) for row in rows)
                path = Path(directory) / f"{split}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                paths.append(path)
            with patch.object(builder, "EVAL_PATHS", tuple(paths)):
                loaded = eval_prompts()
            self.assertTrue(loaded)
            self.assertEqual(loaded, expected)
            self.assertFalse(prompts & loaded)

    def test_overlap_blocks_publication(self):
        prompt = build_rows()[0]["messages"][0]["content"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(json.dumps({"prompt": "  " + prompt.upper() + "  "}) + "\n", encoding="utf-8")
            output = root / "training.jsonl"
            manifest = root / "manifest.json"
            with (patch.object(builder, "EVAL_PATHS", (evaluation,)),
                  patch.object(builder, "OUTPUT", output),
                  patch.object(builder, "MANIFEST", manifest)):
                with self.assertRaisesRegex(ValueError, "exact evaluation overlap"):
                    builder.main()
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())

    def test_held_out_capital_entities_are_absent(self):
        prompts = "\n".join(item["messages"][0]["content"] for item in build_rows())
        for country in ("France", "Peru", "Iceland", "Vietnam", "Morocco", "Finland", "Chile", "Croatia"):
            self.assertNotIn(country, prompts)


if __name__ == "__main__":
    unittest.main()
