import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from configs.default import SpakieConfig
from runtime.processed_data import (
    manifest_path,
    publish_processed_data_manifest,
    validate_processed_data,
)
import scripts.prepare_data as prepare_data
import scripts.run_pipeline as run_pipeline


class ProcessedDataPublicationTests(unittest.TestCase):
    def test_interrupted_merge_preserves_old_arrays_but_invalidates_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir)
            train_path = processed_dir / "train.npy"
            val_path = processed_dir / "val.npy"
            old_train = np.asarray([91, 92, 93], dtype=np.uint16)
            old_val = np.asarray([94, 95], dtype=np.uint16)
            np.save(train_path, old_train)
            np.save(val_path, old_val)
            publish_processed_data_manifest(
                train_path,
                val_path,
                train_tokens=len(old_train),
                val_tokens=len(old_val),
                dtype=np.uint16,
            )
            self.assertTrue(run_pipeline.processed_data_ready(processed_dir)[0])

            shard_paths = [processed_dir / "shard-0.npy", processed_dir / "shard-1.npy"]
            np.save(shard_paths[0], np.asarray([1, 2, 3], dtype=np.uint16))
            np.save(shard_paths[1], np.asarray([4, 5, 6], dtype=np.uint16))

            real_load = np.load
            load_count = 0

            def interrupt_during_copy(*args, **kwargs):
                nonlocal load_count
                load_count += 1
                # Calls 1-2 inspect shard lengths, call 3 copies shard zero,
                # and call 4 begins the second copy after a temp array is
                # already partially populated.
                if load_count == 4:
                    raise KeyboardInterrupt
                return real_load(*args, **kwargs)

            with patch.object(prepare_data.np, "load", side_effect=interrupt_during_copy):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_data.merge_shards(
                        shard_paths,
                        train_path,
                        val_path,
                        train_fraction=0.5,
                        dtype=np.uint16,
                    )

            np.testing.assert_array_equal(np.load(train_path), old_train)
            np.testing.assert_array_equal(np.load(val_path), old_val)
            self.assertFalse(manifest_path(processed_dir).exists())
            ready, reason = run_pipeline.processed_data_ready(processed_dir)
            self.assertFalse(ready)
            self.assertIn("missing completion manifest", reason)
            self.assertEqual(list(processed_dir.glob(".train.npy.*.tmp")), [])
            self.assertEqual(list(processed_dir.glob(".val.npy.*.tmp")), [])

    def test_successful_merge_publishes_manifest_after_both_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_dir = Path(tmpdir)
            shard_paths = [processed_dir / "shard-0.npy", processed_dir / "shard-1.npy"]
            np.save(shard_paths[0], np.asarray([1, 2, 3], dtype=np.uint16))
            np.save(shard_paths[1], np.asarray([4, 5, 6], dtype=np.uint16))
            train_path = processed_dir / "train.npy"
            val_path = processed_dir / "val.npy"

            counts = prepare_data.merge_shards(
                shard_paths,
                train_path,
                val_path,
                train_fraction=0.5,
                dtype=np.uint16,
            )

            self.assertEqual(counts, (3, 3))
            np.testing.assert_array_equal(np.load(train_path), [1, 2, 3])
            np.testing.assert_array_equal(np.load(val_path), [4, 5, 6])
            self.assertTrue(validate_processed_data(processed_dir)[0])
            self.assertTrue(run_pipeline.processed_data_ready(processed_dir)[0])

            payload = json.loads(manifest_path(processed_dir).read_text(encoding="utf-8"))
            self.assertEqual(payload["train"]["tokens"], 3)
            self.assertEqual(payload["val"]["tokens"], 3)


class DeterministicParallelStreamTests(unittest.TestCase):
    def test_parallel_stream_uses_order_preserving_pool_map(self):
        files = [Path("a.jsonl"), Path("b.jsonl"), Path("c.jsonl")]
        results = {
            str(path): {
                "file_path": str(path),
                "file_bytes": index + 1,
                "documents": [("source", 1, True, None, path.stem, None, None)],
            }
            for index, path in enumerate(files)
        }

        class FakePool:
            def __init__(self, *args, **kwargs):
                self.used_ordered_imap = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, function, paths, chunksize):
                self.used_ordered_imap = True
                self.asserted_chunksize = chunksize
                for path in paths:
                    yield results[path]

            def imap_unordered(self, *args, **kwargs):
                raise AssertionError("unordered result consumption breaks exact resume")

            def terminate(self):
                pass

        fake_pool = FakePool()

        class FakeContext:
            def Pool(self, *args, **kwargs):
                return fake_pool

        with patch.object(prepare_data.mp, "get_context", return_value=FakeContext()):
            stream = list(
                prepare_data._doc_stream_parallel(
                    Path("."),
                    files,
                    SpakieConfig(),
                    progress=None,
                    workers=3,
                    dedup_enabled=False,
                    num_perm=8,
                    shingle_size=3,
                )
            )

        self.assertTrue(fake_pool.used_ordered_imap)
        self.assertEqual(fake_pool.asserted_chunksize, 1)
        self.assertEqual([entry[4] for entry in stream], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
