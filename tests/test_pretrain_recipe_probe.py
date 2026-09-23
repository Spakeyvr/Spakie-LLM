import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.probe_pretrain_recipe import SOURCES, batch_from_streams, main, text_key


class RecipeProbeTests(unittest.TestCase):
    def test_source_draws_can_be_independent_of_offset_rng_consumption(self):
        class VirtualStream:
            def __init__(self, length):
                self.length = length
            def __len__(self):
                return self.length
            def __getitem__(self, window):
                return np.zeros(window.stop - window.start, dtype=np.int32)

        small = {s: VirtualStream(4096) for s in SOURCES}
        large = {s: VirtualStream(2**34) for s in SOURCES}
        def draws(streams, separate):
            return [d.tolist() for _, d in batch_from_streams(
                streams, 'baseline', 29, 16, independent_source_rng=separate)]
        # Changing the integer range consumes a different number of RNG bits.
        self.assertNotEqual(draws(small, False), draws(large, False))
        self.assertEqual(draws(small, True), draws(large, True))

    def test_sampling_is_reproducible_and_preserves_shifted_targets(self):
        streams = {source: np.arange(4096, dtype=np.int32) + i * 10000
                   for i, source in enumerate(SOURCES)}
        first = list(batch_from_streams(streams, 'baseline', 17, 4))
        second = list(batch_from_streams(streams, 'baseline', 17, 4))
        for (batch, draws), (again, draws_again) in zip(first, second):
            np.testing.assert_array_equal(batch, again)
            np.testing.assert_array_equal(draws, draws_again)
            np.testing.assert_array_equal(batch[:, 1:] - batch[:, :-1], 1)
            for row, source in zip(batch, draws):
                self.assertEqual(int(row[0]) // 10000, int(source))

    def test_document_key_catches_whitespace_only_duplicates(self):
        self.assertEqual(text_key('A fact.\nMore text.'), text_key(' A  fact. More text. '))
        self.assertNotEqual(text_key('A fact.'), text_key('Another fact.'))

    def test_invalid_budgets_reject_without_creating_outputs(self):
        for flags in [('--steps', '513'), ('--seconds', '181'), ('--lr', 'nan'),
                      ('--scale', '360m', '--steps', '65')]:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'out'
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main(['prepare', '--assets', str(Path(tmp)/'assets'), '--output', str(output), *flags])
                self.assertEqual(exc.exception.code, 2)
                self.assertFalse(output.exists())

    def test_original_asset_outputs_and_existing_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for assets, output in [(Path(tmp), Path(tmp)/'child'), (Path(tmp)/'assets', Path(tmp))]:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main(['prepare', '--assets', str(assets), '--output', str(output)])
                self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
