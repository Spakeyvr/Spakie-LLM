import json
from pathlib import Path
import tempfile
import unittest

from runtime.corpus_contracts import mark_retained_lossy_code, validate_raw_code_provenance


class CorpusContractTests(unittest.TestCase):
    def test_migrating_metadata_cannot_hide_retained_lossy_rows(self):
        progress = {'docs_written': 12, 'python_edu_filter_schema_version': 2}
        mark_retained_lossy_code(progress)
        progress['python_edu_filter_schema_version'] = 4
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'large_corpus/python_edu'
            source.mkdir(parents=True)
            (source/'progress.json').write_text(json.dumps(progress))
            with self.assertRaisesRegex(ValueError, 'cannot repair string literals'):
                validate_raw_code_provenance(tmp)
            validate_raw_code_provenance(tmp, selected_paths=[Path(tmp)/'wikipedia/doc.jsonl'])
        self.assertEqual(progress['docs_written'], 12)

    def test_fresh_and_unmanaged_sources_remain_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'python_edu'
            source.mkdir()
            validate_raw_code_provenance(tmp)
            progress = {'docs_written': 0, 'python_edu_filter_schema_version': 2}
            mark_retained_lossy_code(progress)
            self.assertNotIn('python_edu_requires_fresh_download', progress)
            progress.update(docs_written=12, python_edu_filter_schema_version=4)
            (source/'progress.json').write_text(json.dumps(progress))
            validate_raw_code_provenance(tmp)

    def test_committed_rows_with_stale_zero_count_are_still_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source/'part.jsonl').write_text('{"text":"old code"}\n')
            progress = {'docs_written': 0, 'python_edu_filter_schema_version': 2}
            mark_retained_lossy_code(progress, source_dir=source)
            self.assertTrue(progress['python_edu_requires_fresh_download'])

    def test_both_asset_builders_reject_old_downloads_before_writing(self):
        from configs.default import SpakieConfig
        from scripts.prepare_data import prepare_data
        from tokenizer.train_tokenizer import train_tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'raw/large_corpus/python_edu'
            source.mkdir(parents=True)
            path = source/'progress.json'
            path.write_text(json.dumps({'docs_written': 1, 'python_edu_filter_schema_version': 2}))
            before = path.read_bytes()
            cfg = SpakieConfig(raw_data_dir=str(Path(tmp)/'raw'),
                tokenizer_prefix=str(Path(tmp)/'new/tokenizer'), processed_data_dir=str(Path(tmp)/'new/data'))
            for builder in (train_tokenizer, prepare_data):
                with self.assertRaisesRegex(ValueError, 'old lossy whitespace cleaner'):
                    builder(cfg)
            self.assertFalse((Path(tmp)/'new').exists())
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
