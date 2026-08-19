import json
import math
import numpy as np
import sys
import tempfile
import unittest
import torch
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.default import SpakieConfig, get_preset_config
from model.transformer import SpakieGPT, apply_rotary_emb
import scripts.download_pretrain_corpus as download_pretrain_corpus
import scripts.prepare_data as prepare_data


class FakeTokenizer:
    encode_batch_calls = 0
    encode_batch_threads: list[int] = []
    encode_calls = 0

    def __init__(self, _model_path: str):
        self.vocab_size = 24_576
        self.eos_id = 1

    def encode(self, text: str) -> list[int]:
        self.__class__.encode_calls += 1
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
    def test_nemotron_download_oversamples_for_usable_final_cap(self):
        entry = SpakieConfig().sft_source_limits[
            "nemotron_instruction_following_chat_v3"
        ]
        self.assertEqual(entry["limit"], 10_000)
        self.assertGreater(entry["download_limit"], entry["limit"])

    def test_balanced_corpus_mix_and_fixed_context(self):
        config = SpakieConfig()
        enabled = {
            name: entry for name, entry in config.corpus_source_plan.items()
            if entry.get("enabled", True)
        }
        total = sum(int(entry["target_tokens"]) for entry in enabled.values())
        by_kind: dict[str, int] = {}
        for entry in enabled.values():
            kind = str(entry["kind"])
            by_kind[kind] = by_kind.get(kind, 0) + int(entry["target_tokens"])

        self.assertEqual(config.target_train_tokens, 10_000_000_000)
        self.assertEqual(config.vocab_size, 24_576)
        self.assertEqual(config.max_seq_len, 2_048)
        self.assertAlmostEqual(by_kind["web"] / total, 0.45)
        self.assertAlmostEqual((by_kind["reference"] + by_kind["books"]) / total, 0.15)
        self.assertAlmostEqual(by_kind["math"] / total, 0.15)
        self.assertAlmostEqual(by_kind["code"] / total, 0.15)
        self.assertAlmostEqual(
            (by_kind["technical"] + by_kind["synthetic_education"]) / total,
            0.10,
        )
        self.assertEqual(config.sft_optimizer, "adamw")
        self.assertEqual(config.sft_epochs, 1)

    def test_180m_architecture_ablation_presets(self):
        baseline = get_preset_config("180m")
        gqa4 = get_preset_config("180m_gqa4")
        deep = get_preset_config("180m_deep")
        self.assertEqual(
            (baseline.max_seq_len, gqa4.max_seq_len, deep.max_seq_len),
            (2048, 2048, 2048),
        )
        self.assertEqual(gqa4.n_kv_heads, 4)
        self.assertEqual(gqa4.n_heads % gqa4.n_kv_heads, 0)
        self.assertEqual((deep.n_layers, deep.d_model), (24, 768))
        self.assertLess(deep.swiglu_hidden, baseline.swiglu_hidden)

    def test_default_presets_have_recommended_architecture_and_parameter_counts(self):
        expected = {
            "180m": {
                "shape": (24, 768, 12, 4, 2304),
                "parameters": 184_065_792,
            },
            "300m": {
                "shape": (24, 1024, 16, 4, 3072),
                "parameters": 314_626_048,
            },
        }
        for preset, spec in expected.items():
            with self.subTest(preset=preset):
                config = get_preset_config(preset)
                self.assertEqual(
                    (
                        config.n_layers,
                        config.d_model,
                        config.n_heads,
                        config.n_kv_heads,
                        config.swiglu_hidden,
                    ),
                    spec["shape"],
                )
                self.assertEqual(config.mlp_type, "swiglu")
                self.assertEqual(config.position_encoding, "rope")
                self.assertTrue(config.qk_norm)
                with torch.device("meta"):
                    model = SpakieGPT(config)
                self.assertIsNone(model.pos_emb)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    spec["parameters"],
                )

    def test_300m_keeps_memory_safe_vmap_accumulation_enabled(self):
        config = get_preset_config("300m")
        self.assertTrue(config.pretrain_vmap_accum_step)
        self.assertEqual(config.pretrain_vmap_sync_warmup_steps, 10)
        self.assertEqual(config.pretrain_vmap_group_size, 0)

    def test_rope_rotation_preserves_norm_and_position_zero(self):
        values = torch.arange(2 * 3 * 2 * 8, dtype=torch.float32).reshape(2, 3, 2, 8)
        positions = torch.tensor([[0, 1, 2], [2, 1, 0]])
        rotated = apply_rotary_emb(values, positions, theta=100_000.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated, dim=-1),
            torch.linalg.vector_norm(values, dim=-1),
        )
        torch.testing.assert_close(rotated[0, 0], values[0, 0])
        torch.testing.assert_close(rotated[1, 2], values[1, 2])

    def test_rope_rejects_odd_head_dimensions(self):
        with self.assertRaisesRegex(ValueError, "even attention head dimension"):
            SpakieConfig(
                n_heads=2,
                n_kv_heads=1,
                d_model=14,
                position_encoding="rope",
            )

    def test_code_documents_do_not_use_prose_only_filters(self):
        config = SpakieConfig(min_doc_chars=1, source_min_doc_chars={"python_edu": 1})
        keep, reason = prepare_data.should_keep_document(
            "def add(x, y):\n    return x + y\n\nclass Counter:\n    pass\n",
            config,
            "python_edu",
        )
        self.assertTrue(keep, reason)
        self.assertIsNone(
            prepare_data.language_filter_sample(
                "print('hello')\n" * 100, config, "python_edu"
            )
        )

    def test_math_and_code_use_domain_specific_filter_profiles(self):
        config = SpakieConfig(
            min_doc_chars=1,
            source_min_doc_chars={"fineweb-edu": 1, "finemath": 1},
        )
        repeated_math = (
            "Solve the equation and explain each step.\n"
            + "x + x = 2x\n" * 3
            + "Combine the two matching terms.\n"
            + "Each term has coefficient one.\n"
            + "Add the two coefficients together.\n"
            + "The variable remains unchanged.\n"
            + "The resulting coefficient is two.\n"
            + "Therefore the simplified expression is 2x.\n"
        )
        prose_keep, prose_reason = prepare_data.should_keep_document(
            repeated_math, config, "fineweb-edu"
        )
        math_keep, math_reason = prepare_data.should_keep_document(
            repeated_math, config, "finemath"
        )
        self.assertFalse(prose_keep, prose_reason)
        self.assertTrue(math_keep, math_reason)

    def test_full_corpus_quality_gate_reports_kind_shortfall(self):
        config = SpakieConfig(
            corpus_source_plan={
                "finemath": {
                    "kind": "math",
                    "target_tokens": 100,
                    "target_raw_chars": 400,
                    "enabled": True,
                }
            }
        )
        report = {
            "target_processed_tokens": 100,
            "processed_tokens": 30,
            "source_targets": config.scaled_corpus_source_plan(
                target_processed_tokens=100
            ),
            "source_stats": {"finemath": {"tokens_kept": 30}},
        }
        gate = prepare_data.corpus_quality_gate(report, config)
        self.assertTrue(any("corpus: 30.0%" in item for item in gate["failures"]))
        self.assertTrue(any("math sources: 30.0%" in item for item in gate["failures"]))
        self.assertTrue(any("30.0% of token quota" in item for item in gate["warnings"]))

    def test_source_shortfall_is_advisory_when_kind_and_corpus_are_healthy(self):
        config = SpakieConfig(
            corpus_source_plan={
                "fineweb-edu": {
                    "kind": "web",
                    "target_tokens": 80,
                    "target_raw_chars": 320,
                    "enabled": True,
                },
                "refinedweb": {
                    "kind": "web",
                    "target_tokens": 20,
                    "target_raw_chars": 80,
                    "enabled": True,
                },
            }
        )
        targets = config.scaled_corpus_source_plan(target_processed_tokens=100)
        report = {
            "target_processed_tokens": 100,
            "processed_tokens": 90,
            "source_targets": targets,
            "source_stats": {
                "fineweb-edu": {"tokens_kept": 90},
                "refinedweb": {"tokens_kept": 0},
            },
        }
        gate = prepare_data.corpus_quality_gate(report, config)
        self.assertTrue(gate["passed"], gate["failures"])
        self.assertTrue(any("refinedweb: 0.0%" in item for item in gate["warnings"]))

    def test_fast_resume_finalizes_completed_shards_without_raw_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"
            shards = processed / "shards"
            shards.mkdir(parents=True)
            shard_path = shards / "tokens-00000.npy"
            np.save(shard_path, np.array([2, 3, 4, 5, 6, 7], dtype=np.uint16))
            journal = shards / prepare_data.SHARD_RESUME_JOURNAL
            with journal.open("wb") as handle:
                prepare_data.append_accepted_document(
                    handle,
                    prepare_data.AcceptedDocument("fineweb-edu", 3, 10, 1, ()),
                )
                prepare_data.append_accepted_document(
                    handle,
                    prepare_data.AcceptedDocument("fineweb-edu", 3, 10, 2, ()),
                )

            config = SpakieConfig(
                processed_data_dir=str(processed),
                token_shard_dir=str(shards),
                corpus_report_path=str(processed / "corpus_report.json"),
                target_train_tokens=3,
                train_split_fraction=0.5,
                corpus_source_plan={
                    "fineweb-edu": {
                        "kind": "web",
                        "target_tokens": 6,
                        "target_raw_chars": 24,
                        "enabled": True,
                    }
                },
            )
            source_plan = config.scaled_corpus_source_plan(target_processed_tokens=6)
            report_path = processed / "corpus_report.json"
            report_path.write_text(json.dumps({
                "target_train_tokens": 3,
                "target_processed_tokens": 6,
                "target_tokens_requested": 6,
                "processed_tokens": 6,
                "discovered_files": 1,
                "discovered_raw_bytes": 100,
                "dry_run": False,
                "source_targets": source_plan,
                "source_stats": {"fineweb-edu": {"tokens_kept": 6}},
            }), encoding="utf-8")

            result = prepare_data.try_fast_finalize_resume(
                config=config,
                report_dest=report_path,
                shard_paths=[shard_path],
                resume_journal=journal,
                resume_tokens=6,
                target_tokens=6,
                source_plan=source_plan,
                discovered_files=1,
                discovered_raw_bytes=100,
                tokenizer_provenance={"vocab_size": 24_576},
                preparation_provenance={"schema_version": 4},
                raw_input_provenance={"schema_version": 1},
                max_token_id=7,
                token_dtype=np.uint16,
                enforce_quality_gates=True,
                full_corpus_run=True,
            )

            self.assertIsNotNone(result)
            self.assertTrue(result["resume_fast_finalize"])
            self.assertEqual(np.load(processed / "train.npy").tolist(), [2, 3, 4])
            self.assertEqual(np.load(processed / "val.npy").tolist(), [5, 6, 7])
            self.assertTrue((processed / "processed_data_manifest.json").exists())

    def test_default_source_plan_matches_processed_target(self):
        config = SpakieConfig()
        source_plan = config.scaled_corpus_source_plan()
        self.assertEqual(
            sum(int(entry["target_tokens"]) for entry in source_plan.values()),
            config.target_processed_tokens,
        )

    def test_prepare_data_requires_configured_tokenizer_vocab(self):
        class OldTokenizer:
            vocab_size = 16_384

            def __init__(self, _path):
                pass

        config = SpakieConfig(vocab_size=24_576)
        with (
            patch.object(prepare_data, "SpakieTokenizer", OldTokenizer),
            patch.object(
                prepare_data,
                "tokenizer_contract",
                return_value={"sha256": "old", "vocab_size": 16_384},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Tokenizer vocabulary mismatch"):
                prepare_data.prepare_data(config=config, dry_run=True)
        self.assertEqual(
            config.target_processed_tokens,
            math.ceil(config.target_train_tokens / config.train_split_fraction),
        )
        self.assertEqual(config.pretrain_target_tokens, config.target_train_tokens)

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
                    "The history of computing begins with early mechanical devices that helped people perform "
                    "arithmetic. Over time these machines grew more sophisticated and eventually became the "
                    "programmable computers we recognize today."
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

            with (
                patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer),
                patch.object(
                    prepare_data,
                    "tokenizer_contract",
                    return_value={"sha256": "fake", "vocab_size": 24_576},
                ),
            ):
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
            processed_dir = root / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            existing_train = processed_dir / "train.npy"
            existing_val = processed_dir / "val.npy"
            existing_train.write_bytes(b"pre-existing-train")
            existing_val.write_bytes(b"pre-existing-val")

            config = SpakieConfig(
                raw_data_dir=str(root / "raw"),
                processed_data_dir=str(processed_dir),
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
                patch.object(
                    prepare_data,
                    "tokenizer_contract",
                    return_value={"sha256": "fake", "vocab_size": 24_576},
                ),
                patch.object(prepare_data, "should_keep_document", side_effect=KeyboardInterrupt),
            ):
                report = prepare_data.prepare_data(config=config, dry_run=True, workers=1)

            self.assertTrue(report_path.exists())
            self.assertEqual(report["processed_tokens"], 0)
            self.assertEqual(report["source_stats"]["fineweb-edu"]["documents_seen"], 0)
            # A --dry_run never writes train/val arrays, so an interrupt during
            # a dry run must not delete pre-existing merged arrays from an
            # earlier, completed, non-dry-run invocation.
            self.assertTrue(existing_train.exists())
            self.assertTrue(existing_val.exists())

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
                {"text": "The morning fog rolled across the bay and softened every distant outline."},
                {"text": "She arranged the books on the shelf so each spine caught the afternoon light."},
                {"text": "The morning fog rolled across the bay and softened every distant outline."},
                {"text": "Travelers waited on the platform, watching the slow arrival of the night express."},
                {"text": "He measured the flour twice before adding it to the warm bowl of yeast."},
                {"text": "Children laughed in the courtyard as the kite climbed above the brick chimneys."},
            ]
            sample = raw_dir / "sample.jsonl"
            sample.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            def make_config(name: str) -> SpakieConfig:
                return SpakieConfig(
                    raw_data_dir=str(root / "raw"),
                    processed_data_dir=str(root / name / "processed"),
                    corpus_report_path=str(root / name / "processed" / "corpus_report.json"),
                    token_shard_dir=str(root / name / "processed" / "shards"),
                    target_train_tokens=250,
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

            with (
                patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer),
                patch.object(
                    prepare_data,
                    "tokenizer_contract",
                    return_value={"sha256": "fake", "vocab_size": 24_576},
                ),
            ):
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
                {"text": "The cat sat on the mat."},
                {"text": "Birds fly to the tall tree."},
                {"text": "Sun and stars light the sky."},
                {"text": "He walked home in the rain."},
                {"text": "Books are stacked on the shelf."},
                {"text": "The wind blew across the lake."},
            ]
            # Multiple files exercise the ordered parallel file stream that
            # exact resume relies on, not just deterministic rows within one
            # file. The varying record lengths encourage workers to finish
            # internally out of order while consumption remains ordered.
            for index, row in enumerate(rows):
                sample = raw_dir / f"sample-{index:02d}.jsonl"
                sample.write_text(json.dumps(row) + "\n", encoding="utf-8")

            def make_config(name: str) -> SpakieConfig:
                return SpakieConfig(
                    raw_data_dir=str(root / "raw"),
                    processed_data_dir=str(root / name / "processed"),
                    corpus_report_path=str(root / name / "processed" / "corpus_report.json"),
                    token_shard_dir=str(root / name / "processed" / "shards"),
                    target_train_tokens=100,
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

            with (
                patch.object(prepare_data, "SpakieTokenizer", FakeTokenizer),
                patch.object(
                    prepare_data,
                    "tokenizer_contract",
                    return_value={"sha256": "fake", "vocab_size": 24_576},
                ),
            ):
                full_report = prepare_data.prepare_data(
                    config=make_config("full"),
                    target_tokens=1_053,
                    tokenizer_threads=1,
                    tokenize_batch_size=2,
                    workers=2,
                )
                full_train = np.load(root / "full" / "processed" / "train.npy")
                full_val = np.load(root / "full" / "processed" / "val.npy")

                resume_config = make_config("resume")
                original_should_keep = prepare_data.should_keep_document
                calls = 0

                def interrupt_after_prefix(text, config, source):
                    nonlocal calls
                    calls += 1
                    if calls == 4:
                        raise KeyboardInterrupt
                    return original_should_keep(text, config, source)

                with patch.object(
                    prepare_data,
                    "should_keep_document",
                    side_effect=interrupt_after_prefix,
                ):
                    partial_report = prepare_data.prepare_data(
                        config=resume_config,
                        target_tokens=1_053,
                        tokenizer_threads=1,
                        tokenize_batch_size=1,
                        workers=1,
                    )
                FakeTokenizer.encode_calls = 0
                with patch.object(
                    prepare_data,
                    "compute_minhash_signature",
                    wraps=prepare_data.compute_minhash_signature,
                ) as minhash_mock:
                    resumed_report = prepare_data.prepare_data(
                        config=resume_config,
                        target_tokens=1_053,
                        tokenizer_threads=1,
                        tokenize_batch_size=1,
                        resume=True,
                        workers=1,
                    )
                    resume_minhash_calls = minhash_mock.call_count
                resume_encode_calls = FakeTokenizer.encode_calls
                resumed_train = np.load(root / "resume" / "processed" / "train.npy")
                resumed_val = np.load(root / "resume" / "processed" / "val.npy")

            self.assertGreater(partial_report["processed_tokens"], 0)
            self.assertEqual(full_report["processed_tokens"], resumed_report["processed_tokens"])
            self.assertGreater(resumed_report["resume_existing_shards"], 0)
            self.assertLess(resume_encode_calls, len(rows))
            self.assertLess(resume_minhash_calls, len(rows))
            self.assertEqual(full_train.tolist(), resumed_train.tolist())
            self.assertEqual(full_val.tolist(), resumed_val.tolist())

    def test_pretrain_budget_derives_steps_for_presets(self):
        config_92m = get_preset_config("92m")
        tokens_per_step_92m = (
            config_92m.pretrain_batch_size
            * config_92m.pretrain_grad_accum_steps
            * config_92m.max_seq_len
        )
        self.assertEqual(config_92m.pretrain_tokens_per_step(), tokens_per_step_92m)
        self.assertEqual(
            config_92m.pretrain_max_steps,
            math.ceil(config_92m.pretrain_target_tokens / tokens_per_step_92m),
        )

        config_180m = get_preset_config("180m")
        tokens_per_step_180m = (
            config_180m.pretrain_batch_size
            * config_180m.pretrain_grad_accum_steps
            * config_180m.max_seq_len
        )
        self.assertEqual(config_180m.pretrain_tokens_per_step(), tokens_per_step_180m)
        self.assertEqual(
            config_180m.pretrain_max_steps,
            math.ceil(config_180m.pretrain_target_tokens / tokens_per_step_180m),
        )
        self.assertFalse(config_180m.should_use_pretrain_early_stopping())


if __name__ == "__main__":
    unittest.main()
