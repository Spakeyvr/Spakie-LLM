"""Real SentencePiece regression tests for the pretraining text contract."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from configs.default import SpakieConfig
from tokenizer.train_tokenizer import (
    SpakieTokenizer, iter_sentencepiece_chunks, iter_training_texts, train_tokenizer,
)


class TokenizerWhitespaceTests(unittest.TestCase):
    def test_chunking_is_lossless_at_unicode_and_whitespace_boundaries(self):
        text = '  def café(x):\n\tif x:\n        return "🙂  yes"\n\n  '
        for limit in (4, 7, 23, 4096):
            chunks = list(iter_sentencepiece_chunks(text, max_bytes=limit))
            self.assertEqual(''.join(chunks), text)
            self.assertTrue(all(len(c.encode('utf-8')) <= limit for c in chunks))
        with self.assertRaises(ValueError):
            list(iter_sentencepiece_chunks('🙂', max_bytes=3))

    def test_plain_text_samples_keep_leading_indentation(self):
        text = '    return x\n\treturn y\n'
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'code.txt').write_text(text)
            self.assertEqual(''.join(iter_training_texts(directory)), text)

    def test_trained_tokenizer_roundtrips_nested_code_tabs_and_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw' / 'large_corpus' / 'python_edu'
            raw.mkdir(parents=True)
            examples = [
                f'def function_{i}(x):\n    if x > {i}:\n        return "two  spaces"\n    return {i}\n'
                for i in range(64)
            ]
            (raw / 'data.jsonl').write_text(''.join(json.dumps({'text':s})+'\n' for s in examples))
            config = SpakieConfig(raw_data_dir=str(root/'raw'), tokenizer_prefix=str(root/'tiny'), vocab_size=320)
            # Exercise the real collector/spool/trainer/decoder. Filtering has
            # separate tests; this corpus is intentionally tiny and repetitive.
            with patch('tokenizer.train_tokenizer._clean_tokenizer_texts', return_value=lambda source,text:text):
                train_tokenizer(config, max_sentences=64)
            tokenizer = SpakieTokenizer(str(root/'tiny.model'))
            vocab_rows = (root/'tiny.vocab').read_text().splitlines()
            self.assertEqual(len(vocab_rows), tokenizer.vocab_size)
            for index, row in enumerate(vocab_rows):
                piece, score = row.split('\t')
                self.assertEqual(json.loads('"'+piece+'"'), tokenizer.id_to_piece(index))
                float(score)
            unseen = 'def f(x):\n    if x:\n        return "a  b"\n    return 0\n'
            tabbed = '\tif True:\n\t\tprint("é🙂")\n'
            for text in (unseen, tabbed, '  leading   and trailing  ', '\n\n'):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            compile(tokenizer.decode(tokenizer.encode(unseen)), '<roundtrip>', 'exec')


if __name__ == '__main__':
    unittest.main()
