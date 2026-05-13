import json
import math
import numpy as np
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
    encode_batch_calls = 0
    encode_batch_threads: list[int] = []

    def __init__(self, _model_path: str):
        self.vocab_size = 8192
        self.eos_id = 1

    def encode(self, text: str) -> list[int]:
        tokens = [2 + ((ord(ch) + idx) % 8000) for idx, ch in enumerate(text)]
        return tokens or [2]

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        num_threads: int = -1,
    ) -> list[list[int]]:
        self.__class__.encode_batch_calls += 1
        self.__class__.encode_batch_threads.append(num_threads)
        results = [self.encode(text) for text in texts]
        if add_eos:
            results = [tokens + [self.eos_id] for tokens in results]
        return results


class ScalingConfigTests(unittest.TestCase):
    def test_default_source_plan_matches_processed_target(self):
        config = SpakieConfig()
        source_plan = config.scaled_corpus_source_plan()
        self.assertEqual(
            sum(int(entry["target_tokens"]) for entry in source_plan.values()),
            config.target_processed_tokens,
        )
        self.assertEqual(config.target_train_tokens, 4_000_000_000)
        self.assertEqual(config.pretrain_target_tokens, 4_000_000_000)

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

    def test_prepare_data_ctrl_c_writes_partial_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw" / "large_corpus" / "fineweb-edu"
            raw_dir.mkdir(parents=True, exist_ok=True)
            sample = raw_dir / "sample.jsonl"
            sample.write_text(json.dumps({
                "text": "This document is long enough to enter filtering before interruption."
            }) + "\n", encoding="utf-8")

            report_path = root / "processed" / "corpus_report.json"
            config = SpakieConfig(
                raw_data_dir=str(root / "raw"),
                processed_data_dir=str(root / "processed"),
                corpus_report_path=str(report_path),
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

            with (
                patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer),
                patch.object(prepare_data, "should_keep_document", side_effect=KeyboardInterrupt),
            ):
                report = prepare_data.prepare_data(config=config, dry_run=True, workers=1)

            self.assertTrue(report_path.exists())
            self.assertEqual(report["processed_tokens"], 0)
            self.assertEqual(report["source_stats"]["fineweb-edu"]["documents_seen"], 0)

    def test_token_shard_writer_preserves_token_order_across_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_dir = Path(tmpdir)
            writer = prepare_data.TokenShardWriter(shard_dir, shard_size=5, dtype=np.uint16)

            writer.add([1, 2, 3])
            writer.add([4, 5, 6, 7, 8, 9, 10, 11])
            shard_paths = writer.close()

            tokens = np.concatenate([np.load(path) for path in shard_paths])
            self.assertEqual(tokens.tolist(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            self.assertEqual([np.load(path).shape[0] for path in shard_paths], [5, 5, 1])

    def test_recommended_tokenizer_threads_leaves_headroom_on_18_core_cpu(self):
        self.assertEqual(prepare_data.recommended_tokenizer_threads(18), 16)
        self.assertEqual(prepare_data.recommended_tokenizer_threads(4), 4)

    def test_batched_tokenization_matches_serial_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw" / "large_corpus" / "fineweb-edu"
            raw_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"text": "alpha beta gamma"},
                {"text": "delta epsilon zeta"},
                {"text": "alpha beta gamma"},
                {"text": "eta theta iota"},
                {"text": "kappa lambda mu"},
                {"text": "nu xi omicron"},
            ]
            sample = raw_dir / "sample.jsonl"
            sample.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            def make_config(name: str) -> SpakieConfig:
                return SpakieConfig(
                    raw_data_dir=str(root / "raw"),
                    processed_data_dir=str(root / name / "processed"),
                    corpus_report_path=str(root / name / "processed" / "corpus_report.json"),
                    token_shard_dir=str(root / name / "processed" / "shards"),
                    target_train_tokens=10,
                    min_doc_chars=1,
                    token_shard_size=7,
                    source_min_doc_chars={"fineweb-edu": 1},
                    corpus_source_plan={
                        "fineweb-edu": {
                            "kind": "web",
                            "target_tokens": 1_053,
                            "target_raw_chars": 4_212,
                            "enabled": True,
                        }
                    },
                )

            with patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer):
                FakeTokenizer.encode_batch_calls = 0
                FakeTokenizer.encode_batch_threads = []
                serial_report = prepare_data.prepare_data(
                    config=make_config("serial"),
                    target_tokens=1_053,
                    tokenizer_threads=1,
                    tokenize_batch_size=1,
                )
                serial_train = np.load(root / "serial" / "processed" / "train.npy")
                serial_val = np.load(root / "serial" / "processed" / "val.npy")

                batched_report = prepare_data.prepare_data(
                    config=make_config("batched"),
                    target_tokens=1_053,
                    tokenizer_threads=4,
                    tokenize_batch_size=3,
                )
                batched_train = np.load(root / "batched" / "processed" / "train.npy")
                batched_val = np.load(root / "batched" / "processed" / "val.npy")

            self.assertEqual(serial_report["processed_tokens"], batched_report["processed_tokens"])
            self.assertEqual(serial_train.tolist(), batched_train.tolist())
            self.assertEqual(serial_val.tolist(), batched_val.tolist())
            self.assertGreater(FakeTokenizer.encode_batch_calls, 0)
            self.assertIn(4, FakeTokenizer.encode_batch_threads)

    def test_prepare_data_resume_appends_existing_token_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw" / "large_corpus" / "fineweb-edu"
            raw_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"text": "alpha beta gamma"},
                {"text": "delta epsilon zeta"},
                {"text": "eta theta iota"},
                {"text": "kappa lambda mu"},
                {"text": "nu xi omicron"},
                {"text": "pi rho sigma"},
            ]
            sample = raw_dir / "sample.jsonl"
            sample.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            def make_config(name: str) -> SpakieConfig:
                return SpakieConfig(
                    raw_data_dir=str(root / "raw"),
                    processed_data_dir=str(root / name / "processed"),
                    corpus_report_path=str(root / name / "processed" / "corpus_report.json"),
                    token_shard_dir=str(root / name / "processed" / "shards"),
                    target_train_tokens=10,
                    min_doc_chars=1,
                    token_shard_size=11,
                    source_min_doc_chars={"fineweb-edu": 1},
                    corpus_source_plan={
                        "fineweb-edu": {
                            "kind": "web",
                            "target_tokens": 1_053,
                            "target_raw_chars": 4_212,
                            "enabled": True,
                        }
                    },
                )

            with patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer):
                full_report = prepare_data.prepare_data(
                    config=make_config("full"),
                    target_tokens=1_053,
                    tokenizer_threads=1,
                    tokenize_batch_size=2,
                )
                full_train = np.load(root / "full" / "processed" / "train.npy")
                full_val = np.load(root / "full" / "processed" / "val.npy")

                resume_config = make_config("resume")
                partial_report = prepare_data.prepare_data(
                    config=resume_config,
                    target_tokens=40,
                    tokenizer_threads=1,
                    tokenize_batch_size=2,
                )
                resumed_report = prepare_data.prepare_data(
                    config=resume_config,
                    target_tokens=1_053,
                    tokenizer_threads=1,
                    tokenize_batch_size=2,
                    resume=True,
                )
                resumed_train = np.load(root / "resume" / "processed" / "train.npy")
                resumed_val = np.load(root / "resume" / "processed" / "val.npy")

            self.assertGreater(partial_report["processed_tokens"], 0)
            self.assertEqual(full_report["processed_tokens"], resumed_report["processed_tokens"])
            self.assertGreater(resumed_report["resume_existing_shards"], 0)
            self.assertEqual(full_train.tolist(), resumed_train.tolist())
            self.assertEqual(full_val.tolist(), resumed_val.tolist())

    def test_pretrain_budget_derives_steps_for_presets(self):
        config_92m = get_preset_config("92m")
        self.assertEqual(config_92m.pretrain_tokens_per_step(), 32_768)   # 64 * 1 * 512
        self.assertEqual(config_92m.pretrain_max_steps, math.ceil(4_000_000_000 / 32_768))

        config_180m = get_preset_config("180m")
        self.assertEqual(config_180m.pretrain_tokens_per_step(), 98_304)  # 96 * 2 * 512
        self.assertEqual(config_180m.pretrain_max_steps, math.ceil(4_000_000_000 / 98_304))
        self.assertFalse(config_180m.should_use_pretrain_early_stopping())


if __name__ == "__main__":
    unittest.main()
