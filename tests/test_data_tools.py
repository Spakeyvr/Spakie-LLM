import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.default import SpakieConfig
import scripts.build_targeted_data as build_targeted_data
import scripts.download_pretrain_corpus as download_pretrain_corpus
import scripts.prepare_data as prepare_data
import scripts.scrape_dictionary as scrape_dictionary
import scripts.scrape_wiki as scrape_wiki
import tokenizer.train_tokenizer as train_tokenizer


class CompactResumeIndexTests(unittest.TestCase):
    def test_cosmopedia_oversized_legacy_ids_are_archived_without_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            (source_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "source": "cosmopedia_v2",
                        "docs_written": 2,
                        "chars_written": 1000,
                        "estimated_tokens": 250,
                        "shard_index": 0,
                        "hf_rows_seen": 10,
                        "site_pages": {},
                        "arxiv_offsets": {},
                    }
                ),
                encoding="utf-8",
            )
            legacy = source_dir / "seen_ids.txt"
            legacy.write_text("a raw prompt\nwith embedded lines\n", encoding="utf-8")
            budget = download_pretrain_corpus.SourceBudget(
                "cosmopedia_v2", "synthetic_education", 10_000, 100, 2500
            )

            with patch.object(download_pretrain_corpus, "LEGACY_SEEN_IDS_MAX_BYTES", 1):
                state = download_pretrain_corpus.SourceState(
                    source_dir,
                    budget,
                    resume=True,
                    config=SpakieConfig(),
                )

            self.assertFalse(legacy.exists())
            self.assertTrue((source_dir / "seen_ids.legacy-raw.txt").exists())
            self.assertEqual(state.seen_ids, set())

            record = {
                "id": "prompt\nthat would previously span lines",
                "title": "",
                "url": "",
                "text": "This is ordinary English prose. " * 30,
                "meta": {},
            }
            self.assertTrue(state.accept(record, english_only=False))
            state.close()

            compact_ids = legacy.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(compact_ids), 1)
            self.assertEqual(len(compact_ids[0]), 32)
            int(compact_ids[0], 16)

            resumed = download_pretrain_corpus.SourceState(
                source_dir,
                budget,
                resume=True,
                config=SpakieConfig(),
            )
            self.assertFalse(resumed.accept(dict(record), english_only=False))
            resumed.close()

    def test_pretrain_downloader_returns_failure_for_failed_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpakieConfig(
                large_corpus_dir=str(Path(tmpdir) / "corpus"),
                target_train_tokens=100,
                corpus_source_plan={
                    "fineweb-edu": {
                        "kind": "web",
                        "target_tokens": 106,
                        "target_raw_chars": 424,
                        "enabled": True,
                    }
                },
            )
            argv = [
                "download_pretrain_corpus.py",
                "--sources",
                "fineweb-edu",
                "--workers",
                "1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_pretrain_corpus, "SpakieConfig", return_value=config),
                patch.object(
                    download_pretrain_corpus,
                    "run_source",
                    side_effect=RuntimeError("network failed"),
                ),
            ):
                result = download_pretrain_corpus.main()

        self.assertEqual(result, 1)


class TargetedDataSplitTests(unittest.TestCase):
    def test_generated_basic_eval_has_no_exact_training_prompt(self):
        facts = build_targeted_data.fact_rows()
        train_rows = build_targeted_data.build_sft(facts, seed=42)
        train_prompts = build_targeted_data.training_user_prompts(train_rows)
        eval_rows = build_targeted_data.build_basic_eval(
            facts,
            excluded_prompts=train_prompts,
        )

        self.assertTrue(eval_rows)
        self.assertTrue(train_prompts.isdisjoint(row["question"] for row in eval_rows))


class TokenizerSamplingTests(unittest.TestCase):
    def test_training_texts_interleave_sources_deterministically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            early = root / "large_corpus" / "aaa" / "data.jsonl"
            late = root / "large_corpus" / "zzz" / "data.jsonl"
            early.parent.mkdir(parents=True)
            late.parent.mkdir(parents=True)
            early.write_text(
                "".join(json.dumps({"text": f"early-{i}"}) + "\n" for i in range(6)),
                encoding="utf-8",
            )
            late.write_text(
                "".join(json.dumps({"text": f"late-{i}"}) + "\n" for i in range(6)),
                encoding="utf-8",
            )

            first = list(train_tokenizer.iter_training_texts(str(root)))[:6]
            second = list(train_tokenizer.iter_training_texts(str(root)))[:6]

        self.assertEqual(first, second)
        self.assertEqual(first, ["early-0", "late-0", "early-1", "late-1", "early-2", "late-2"])


class CanonicalLanguageFilterTests(unittest.TestCase):
    def test_prepare_data_filters_long_non_english_document(self):
        class FakeTokenizer:
            vocab_size = 100
            eos_id = 1

            def __init__(self, _path):
                pass

            def encode(self, text):
                return [2] * len(text)

            def encode_batch(self, texts, **_kwargs):
                return [[2] * len(text) + [self.eos_id] for text in texts]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw" / "large_corpus" / "fineweb-edu"
            raw.mkdir(parents=True)
            (raw / "docs.jsonl").write_text(
                json.dumps({"text": "KEEP " + "ordinary prose " * 40})
                + "\n"
                + json.dumps({"text": "DROP " + "foreign prose " * 40})
                + "\n",
                encoding="utf-8",
            )
            config = SpakieConfig(
                raw_data_dir=str(root / "raw"),
                processed_data_dir=str(root / "processed"),
                corpus_report_path=str(root / "processed" / "report.json"),
                token_shard_dir=str(root / "processed" / "shards"),
                tokenizer_prefix=str(root / "tokenizer"),
                token_shard_size=100,
                target_train_tokens=10,
                min_doc_chars=1,
                source_min_doc_chars={"fineweb-edu": 1},
                corpus_source_plan={
                    "fineweb-edu": {
                        "kind": "web",
                        "target_tokens": 10_000,
                        "target_raw_chars": 40_000,
                        "enabled": True,
                    }
                },
            )
            with (
                patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer),
                patch.object(prepare_data, "should_keep_document", return_value=(True, "")),
                patch.object(
                    prepare_data,
                    "is_probably_english",
                    side_effect=lambda text, _config: text.startswith("KEEP"),
                ),
            ):
                report = prepare_data.prepare_data(
                    config=config,
                    target_tokens=10_000,
                    dedup=False,
                    workers=1,
                    tokenizer_threads=1,
                )

        stats = report["source_stats"]["fineweb-edu"]
        self.assertEqual(stats["documents_kept"], 1)
        self.assertEqual(stats["drop_reasons"]["non_english"], 1)


class ScraperResumeTests(unittest.TestCase):
    def test_wiki_failed_fetch_is_not_marked_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "wiki"
            progress_path = output / ".progress.json"
            with (
                patch.object(scrape_wiki, "OUTPUT_DIR", str(output)),
                patch.object(scrape_wiki, "PROGRESS_FILE", str(progress_path)),
                patch.object(scrape_wiki, "CURATED_ARTICLES", ["retry-me", "success"]),
                patch.object(scrape_wiki, "fetch_article", side_effect=[None, "valid " * 100]),
                patch.object(scrape_wiki.time, "sleep"),
            ):
                scrape_wiki.scrape()

            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["done"], ["success"])

    def test_dictionary_failed_fetch_stays_in_retry_queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "dictionary"
            output.mkdir()
            progress_path = output / ".progress.json"
            queue_path = output / ".queue.json"
            queue_path.write_text(json.dumps(["retry-me", "success"]), encoding="utf-8")
            with (
                patch.object(scrape_dictionary, "OUTPUT_DIR", str(output)),
                patch.object(scrape_dictionary, "PROGRESS_FILE", str(progress_path)),
                patch.object(scrape_dictionary, "QUEUE_FILE", str(queue_path)),
                patch.object(scrape_dictionary.time, "sleep"),
                patch.object(
                    scrape_dictionary,
                    "fetch_definition",
                    side_effect=[None, "definition " * 20],
                ),
            ):
                scrape_dictionary.scrape()

            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["scraped"], ["success"])
            self.assertEqual(progress["retry"], ["retry-me"])
            self.assertEqual(progress["index"], 2)

            with (
                patch.object(scrape_dictionary, "OUTPUT_DIR", str(output)),
                patch.object(scrape_dictionary, "PROGRESS_FILE", str(progress_path)),
                patch.object(scrape_dictionary, "QUEUE_FILE", str(queue_path)),
                patch.object(scrape_dictionary.time, "sleep"),
                patch.object(scrape_dictionary, "fetch_definition", return_value="retried " * 20),
            ):
                scrape_dictionary.scrape()

            retried = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertCountEqual(retried["scraped"], ["success", "retry-me"])
            self.assertEqual(retried["retry"], [])


if __name__ == "__main__":
    unittest.main()
