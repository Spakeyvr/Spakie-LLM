import _thread
import gzip
import io
import json
import sys
import tempfile
import threading
import time
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
    def test_download_progress_uses_compact_rate_and_readable_eta(self):
        self.assertEqual(
            download_pretrain_corpus.AcceptedRateMonitor._format_rate(4_410_000),
            "4.41M",
        )
        self.assertEqual(
            download_pretrain_corpus.AcceptedRateMonitor._format_duration(4_980),
            "1h 23min",
        )
        self.assertEqual(
            download_pretrain_corpus.AcceptedRateMonitor._format_duration(185),
            "3min 5s",
        )

    def test_interrupt_exit_flushes_then_terminates_without_thread_shutdown_wait(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(download_pretrain_corpus.sys, "stdout", stdout),
            patch.object(download_pretrain_corpus.sys, "stderr", stderr),
            patch.object(download_pretrain_corpus.os, "_exit") as hard_exit,
        ):
            download_pretrain_corpus.exit_process(130)

        hard_exit.assert_called_once_with(130)

    def test_ctrl_c_returns_without_waiting_for_blocked_source_worker(self):
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
            release_worker = threading.Event()
            timer = threading.Timer(0.05, _thread.interrupt_main)
            argv = [
                "download_pretrain_corpus.py", "--sources", "fineweb-edu", "--workers", "1"
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_pretrain_corpus, "SpakieConfig", return_value=config),
                patch.object(download_pretrain_corpus, "INTERRUPT_GRACE_SECONDS", 0.05),
                patch.object(
                    download_pretrain_corpus,
                    "run_source",
                    side_effect=lambda *args, **kwargs: release_worker.wait(5),
                ),
            ):
                timer.start()
                started = time.monotonic()
                try:
                    result = download_pretrain_corpus.main()
                finally:
                    release_worker.set()
                    timer.cancel()
                    download_pretrain_corpus.STOP_EVENT.clear()

        self.assertEqual(result, 130)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_near_complete_legacy_cursor_is_deferred_to_direct_source(self):
        budget = download_pretrain_corpus.SourceBudget(
            "fineweb-edu", "web", 4_000_000_000, 0, 1_000_000_000
        )
        legacy = {"hf_rows_seen": 4_000_000, "estimated_tokens": 998_000_000}
        self.assertTrue(download_pretrain_corpus.should_defer_legacy_cursor(legacy, budget))

        large_tail = {"hf_rows_seen": 4_000_000, "estimated_tokens": 900_000_000}
        self.assertFalse(
            download_pretrain_corpus.should_defer_legacy_cursor(large_tail, budget)
        )

        direct = {
            "hf_rows_seen": 4_000_000,
            "estimated_tokens": 998_000_000,
            "hf_stream_state": {"index": 4_000_000},
        }
        self.assertFalse(download_pretrain_corpus.should_defer_legacy_cursor(direct, budget))

    def test_token_budget_remains_authoritative_after_char_estimate_is_reached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = download_pretrain_corpus.SourceState(
                Path(tmpdir),
                download_pretrain_corpus.SourceBudget("arxiv", "technical", 400, 0, 100),
                resume=False,
            )
            state.progress["chars_written"] = 400
            state.progress["estimated_tokens"] = 99
            self.assertFalse(state.should_stop())
            state.progress["estimated_tokens"] = 100
            self.assertTrue(state.should_stop())

    def test_nested_hf_fields_are_resolved(self):
        record = {"metadata": {"url": "https://example.test/item"}}
        self.assertEqual(
            download_pretrain_corpus.pick_first(record, ("missing", "metadata.url")),
            "https://example.test/item",
        )

    def test_python_edu_materializes_public_s3_blob(self):
        compressed = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed, mode="wb") as handle:
            handle.write(b"def answer():\n    return 42\n")
        payload = compressed.getvalue()

        class FakeResponse:
            status_code = 200
            reason = "OK"
            content = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

        class FakeSession:
            def get(self, url, **kwargs):
                self.url = url
                self.kwargs = kwargs
                return FakeResponse()

        session = FakeSession()
        with patch.object(download_pretrain_corpus, "_http_session", return_value=session):
            row = download_pretrain_corpus.materialize_python_edu_row({"blob_id": "abc123"})

        self.assertEqual(row["text"], "def answer():\n    return 42\n")
        self.assertEqual(
            session.url,
            "https://softwareheritage.s3.amazonaws.com/content/abc123",
        )
        self.assertEqual(session.kwargs, {"timeout": (10, 60)})

    def test_python_edu_prefilters_short_and_seen_rows_before_content_fetch(self):
        class FakeStream:
            def __init__(self, rows):
                self.rows = rows
                self.index = 0

            def __iter__(self):
                while self.index < len(self.rows):
                    row = self.rows[self.index]
                    self.index += 1
                    yield row

            def state_dict(self):
                return {"index": self.index}

            def load_state_dict(self, state):
                self.index = int(state["index"])

        rows = [
            {"blob_id": "short", "path": "/short.py", "length_bytes": 120},
            {"blob_id": "seen", "path": "/seen.py", "length_bytes": 800},
            {"blob_id": "fetch", "path": "/fetch.py", "length_bytes": 800},
        ]
        variant = {
            "path": "fake/python-edu",
            "name": "python-edu",
            "split": "train",
        }
        fetched: list[str] = []

        def materialize(row):
            fetched.append(row["blob_id"])
            return {**row, "text": "def useful_example():\n    return 42\n" * 30}

        with tempfile.TemporaryDirectory() as tmpdir:
            state = download_pretrain_corpus.SourceState(
                Path(tmpdir),
                download_pretrain_corpus.SourceBudget(
                    "python_edu", "code", 100_000, 0, 100_000
                ),
                resume=False,
            )
            state.seen_ids.add(state.identity_keys("seen", "", "")[0])
            stream = FakeStream(rows)
            with (
                patch.object(
                    download_pretrain_corpus,
                    "load_hf_stream",
                    return_value=(stream, variant, 0),
                ),
                patch.object(
                    download_pretrain_corpus,
                    "materialize_python_edu_row",
                    side_effect=materialize,
                ),
            ):
                download_pretrain_corpus.ingest_hf_source(
                    "python_edu",
                    state,
                    english_only=False,
                    item_workers=2,
                )
            state.close()

            written = [
                json.loads(line)["id"]
                for shard in sorted(Path(tmpdir).glob("shard-*.jsonl"))
                for line in shard.read_text().splitlines()
            ]

        self.assertEqual(fetched, ["fetch"])
        self.assertEqual(written, ["fetch"])
        self.assertEqual(state.progress["hf_rows_seen"], 3)
        self.assertEqual(state.progress["python_edu_prefiltered_rows"], 2)

    def test_completed_source_is_skipped_before_resume_indexes_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "corpus"
            source_dir = root / "fineweb-edu"
            source_dir.mkdir(parents=True)
            (source_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "source": "fineweb-edu",
                        "docs_written": 1,
                        "chars_written": 424,
                        "estimated_tokens": 106,
                        "shard_index": 1,
                        "hf_rows_seen": 1,
                    }
                ),
                encoding="utf-8",
            )
            config = SpakieConfig(
                large_corpus_dir=str(root),
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
                "--resume",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(download_pretrain_corpus, "SpakieConfig", return_value=config),
                patch.object(download_pretrain_corpus, "run_source") as run_source,
            ):
                result = download_pretrain_corpus.main()

        self.assertEqual(result, 0)
        run_source.assert_not_called()

    def test_non_resume_deletes_existing_source_files_before_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "corpus"
            source_dir = root / "fineweb-edu"
            source_dir.mkdir(parents=True)
            stale_files = (
                "shard-00000.jsonl",
                "shard-00000.manifest.json",
                "progress.json",
                "seen_ids.txt",
                "seen_urls.txt",
                "seen_titles.txt",
            )
            for name in stale_files:
                (source_dir / name).write_text("stale", encoding="utf-8")

            config = SpakieConfig(
                large_corpus_dir=str(root),
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
                "--max_docs",
                "1",
            ]

            def assert_source_was_reset(*_args, **_kwargs):
                self.assertEqual(list(source_dir.iterdir()), [])
                return 0

            with (
                patch.object(sys, "argv", argv),
                patch.object(download_pretrain_corpus, "SpakieConfig", return_value=config),
                patch.object(
                    download_pretrain_corpus,
                    "run_source",
                    side_effect=assert_source_was_reset,
                ) as run_source,
            ):
                result = download_pretrain_corpus.main()

        self.assertEqual(result, 0)
        run_source.assert_called_once()

    def test_hf_stream_state_resumes_without_replaying_prior_rows(self):
        class FakeStream:
            def __init__(self):
                self.rows = [
                    {"id": "first", "text": "First useful document. " * 40},
                    {"id": "second", "text": "Second useful document. " * 40},
                ]
                self.index = 0
                self.loaded_state = None

            def __iter__(self):
                while self.index < len(self.rows):
                    row = self.rows[self.index]
                    self.index += 1
                    yield row

            def state_dict(self):
                return {"index": self.index}

            def load_state_dict(self, state):
                self.loaded_state = dict(state)
                self.index = int(state["index"])

        variant = {"path": "fake/fineweb", "name": None, "split": "train"}
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            first_budget = download_pretrain_corpus.SourceBudget(
                "fineweb-edu", "web", 10_000, 1, 10_000
            )
            first_state = download_pretrain_corpus.SourceState(source_dir, first_budget, resume=False)
            first_stream = FakeStream()
            with patch.object(
                download_pretrain_corpus,
                "load_hf_stream",
                return_value=(first_stream, variant, 0),
            ):
                download_pretrain_corpus.ingest_hf_source(
                    "fineweb-edu", first_state, english_only=False, hf_workers=1
                )
            first_state.close()

            second_budget = download_pretrain_corpus.SourceBudget(
                "fineweb-edu", "web", 20_000, 2, 20_000
            )
            resumed_state = download_pretrain_corpus.SourceState(source_dir, second_budget, resume=True)
            resumed_stream = FakeStream()
            with patch.object(
                download_pretrain_corpus,
                "load_hf_stream",
                return_value=(resumed_stream, variant, 0),
            ):
                download_pretrain_corpus.ingest_hf_source(
                    "fineweb-edu", resumed_state, english_only=False, hf_workers=1
                )
            resumed_state.close()

            rows = []
            for shard in sorted(source_dir.glob("shard-*.jsonl")):
                rows.extend(json.loads(line) for line in shard.read_text().splitlines())

        self.assertEqual(resumed_stream.loaded_state, {"index": 1})
        self.assertEqual([row["id"] for row in rows], ["first", "second"])

    def test_parallel_hf_shards_resume_from_committed_worker_states(self):
        class FakeShard:
            def __init__(self, rows):
                self.rows = rows
                self.index = 0

            def __iter__(self):
                while self.index < len(self.rows):
                    row = self.rows[self.index]
                    self.index += 1
                    yield row

            def state_dict(self):
                return {"index": self.index}

            def load_state_dict(self, state):
                self.index = int(state["index"])

        class FakeDataset:
            num_shards = 2

            def __init__(self):
                self.rows = [
                    {"id": f"doc-{index}", "text": f"Useful document {index}. " * 40}
                    for index in range(4)
                ]

            def shard(self, num_shards, index, contiguous=False):
                self.assertion = (num_shards, contiguous)
                return FakeShard(self.rows[index::num_shards])

        variant = {"path": "fake/fineweb", "name": None, "split": "train"}
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            first = download_pretrain_corpus.SourceState(
                source_dir,
                download_pretrain_corpus.SourceBudget("fineweb-edu", "web", 100_000, 2, 100_000),
                resume=False,
            )
            with patch.object(
                download_pretrain_corpus,
                "load_hf_stream",
                return_value=(FakeDataset(), variant, 0),
            ):
                download_pretrain_corpus.ingest_hf_source(
                    "fineweb-edu", first, english_only=False, hf_workers=2
                )
            first.close()

            resumed = download_pretrain_corpus.SourceState(
                source_dir,
                download_pretrain_corpus.SourceBudget("fineweb-edu", "web", 200_000, 4, 200_000),
                resume=True,
            )
            with patch.object(
                download_pretrain_corpus,
                "load_hf_stream",
                return_value=(FakeDataset(), variant, 0),
            ):
                download_pretrain_corpus.ingest_hf_source(
                    "fineweb-edu", resumed, english_only=False, hf_workers=2
                )
            resumed.close()

            ids = []
            for shard in sorted(source_dir.glob("shard-*.jsonl")):
                ids.extend(json.loads(line)["id"] for line in shard.read_text().splitlines())

        self.assertEqual(sorted(ids), ["doc-0", "doc-1", "doc-2", "doc-3"])
        self.assertEqual(len(ids), len(set(ids)))

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

    def test_sentencepiece_chunks_never_exceed_byte_limit(self):
        text = ("alpha beta gamma delta " * 20) + ("é" * 30)
        chunks = list(train_tokenizer.iter_sentencepiece_chunks(text, max_bytes=40))
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 40 for chunk in chunks))
        self.assertIn("é", "".join(chunks))


class PretrainCleaningTests(unittest.TestCase):
    def test_code_normalization_preserves_indentation(self):
        text = "def outer():\r\n    if True:\r\n        return 1   \r\n"
        cleaned = download_pretrain_corpus.normalize_text(
            text, preserve_indentation=True
        )
        self.assertEqual(cleaned, "def outer():\n    if True:\n        return 1")

    def test_structured_cleaning_preserves_math_code_and_indexing(self):
        text = "  def f(x):\n    return values[1] if x < y else values[2]  "
        cleaned = prepare_data.clean_text(text, "python_edu")
        self.assertIn("    return values[1]", cleaned)
        self.assertIn("x < y", cleaned)

    def test_wikipedia_cleaning_removes_known_markup_and_citations(self):
        cleaned = prepare_data.clean_text(
            "<p>A fact [12]</p> followed by x < y.", "wikipedia_snapshot"
        )
        self.assertEqual(cleaned, "A fact followed by x < y.")

    def test_minhash_result_is_independent_of_block_size(self):
        text = " ".join(f"word{i % 97}" for i in range(20_000))
        with patch.object(prepare_data, "_MINHASH_BLOCK", 31):
            small = prepare_data.compute_minhash_signature(
                text, num_perm=32, shingle_size=5
            )
        with patch.object(prepare_data, "_MINHASH_BLOCK", 4096):
            large = prepare_data.compute_minhash_signature(
                text, num_perm=32, shingle_size=5
            )
        self.assertEqual(small.tolist(), large.tolist())

    def test_python_edu_titles_are_namespaced_by_repository(self):
        variant = download_pretrain_corpus.HF_DATASETS["python_edu"]["variants"][0]
        row = {
            "repo_name": "owner/repo",
            "path": "src/main.py",
            "text": "print('hello')",
        }
        formatted = download_pretrain_corpus.format_hf_record(
            "python_edu", row, variant
        )
        self.assertEqual(formatted["title"], "owner/repo:src/main.py")


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
                + json.dumps({"text": "KEEP " + "different english text " * 35})
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
                target_train_tokens=500,
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
                patch.object(
                    prepare_data,
                    "tokenizer_contract",
                    return_value={"sha256": "fake", "vocab_size": 8192},
                ),
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
        self.assertEqual(stats["documents_kept"], 2)
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
