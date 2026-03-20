import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.default import SpakieConfig, get_preset_config
import scripts.download_pretrain_corpus as download_pretrain_corpus
import scripts.prepare_data as prepare_data


class FakeTokenizer:
    def __init__(self, _model_path: str):
        self.vocab_size = 8192
        self.eos_id = 1

    def encode(self, text: str) -> list[int]:
        return [1] * max(1, len(text) // 8)


class ScalingConfigTests(unittest.TestCase):
    def test_default_source_plan_matches_processed_target(self):
        config = SpakieConfig()
        source_plan = config.scaled_corpus_source_plan()
        self.assertEqual(
            sum(int(entry["target_tokens"]) for entry in source_plan.values()),
            config.target_processed_tokens,
        )
        self.assertEqual(config.target_train_tokens, 2_000_000_000)
        self.assertEqual(config.pretrain_target_tokens, 2_000_000_000)

    def test_parse_sources_all_and_aliases(self):
        config = SpakieConfig()
        self.assertEqual(
            download_pretrain_corpus.parse_sources("all", config),
            [
                source_name
                for source_name, plan in config.corpus_source_plan.items()
                if plan.get("enabled", True)
            ],
        )
        self.assertEqual(
            download_pretrain_corpus.parse_sources("fineweb-edu,wikipedia", config),
            ["fineweb-edu", "wikipedia_snapshot"],
        )

    def test_prepare_data_dry_run_reports_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw" / "large_corpus" / "fineweb-edu"
            raw_dir.mkdir(parents=True, exist_ok=True)
            sample = raw_dir / "sample.jsonl"
            sample.write_text(json.dumps({
                "text": (
                    "This is a clean test document repeated enough times to exceed the minimum threshold. "
                    "This is a clean test document repeated enough times to exceed the minimum threshold."
                )
            }) + "\n", encoding="utf-8")

            config = SpakieConfig(
                raw_data_dir=str(root / "raw"),
                processed_data_dir=str(root / "processed"),
                corpus_report_path=str(root / "processed" / "corpus_report.json"),
                token_shard_dir=str(root / "processed" / "shards"),
                target_train_tokens=100,
                min_doc_chars=1,
                source_min_doc_chars={"fineweb-edu": 1},
                corpus_source_plan={
                    "fineweb-edu": {
                        "kind": "web",
                        "target_tokens": 106,
                        "target_raw_chars": 424,
                        "enabled": True,
                    }
                },
            )

            with patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer):
                report = prepare_data.prepare_data(config=config, dry_run=True)

            self.assertEqual(report["target_train_tokens"], 100)
            self.assertEqual(report["target_processed_tokens"], math.ceil(100 / 0.95))
            self.assertGreater(report["processed_tokens"], 0)
            self.assertIn("fineweb-edu", report["source_targets"])
            self.assertGreater(report["source_stats"]["fineweb-edu"]["target_tokens"], 0)
            self.assertGreaterEqual(report["source_stats"]["fineweb-edu"]["completion_ratio"], 0.0)

    def test_pretrain_budget_derives_steps_for_presets(self):
        config_92m = get_preset_config("92m")
        self.assertEqual(config_92m.pretrain_tokens_per_step(), 32_768)
        self.assertEqual(config_92m.pretrain_max_steps, math.ceil(2_000_000_000 / 32_768))

        config_180m = get_preset_config("180m")
        self.assertEqual(config_180m.pretrain_tokens_per_step(), 16_384)
        self.assertEqual(config_180m.pretrain_max_steps, math.ceil(2_000_000_000 / 16_384))
        self.assertFalse(config_180m.should_use_pretrain_early_stopping())


if __name__ == "__main__":
    unittest.main()
