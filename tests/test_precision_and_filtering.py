"""Regression checks motivated by the trained 92M checkpoint diagnostics."""
import unittest
import numpy as np
import torch
from configs.default import SpakieConfig
from inference.generation_utils import apply_repetition_penalty
from scripts.prepare_data import _dup_ngram_char_share_from_words, should_keep_document
from model.transformer import SpakieGPT
from tests.test_mlx_parity import _skip_if_no_mlx


class RepetitionTests(unittest.TestCase):
    def test_probability_penalty_is_invariant_to_common_offset(self):
        a = np.array([1., 0., -1.], dtype=np.float64)
        b = a - 250
        apply_repetition_penalty(a, [0], 1.2)
        apply_repetition_penalty(b, [0], 1.2)
        np.testing.assert_allclose(a - a.max(), b - b.max(), atol=1e-12)
        self.assertAlmostEqual(np.exp(a[0] - a[1]), np.e / 1.2)

    def test_invalid_penalties_are_rejected(self):
        for p in [float('nan'), float('inf'), 0, -1]:
            with self.assertRaises(ValueError):
                apply_repetition_penalty(np.ones(2), [0], p)

    def test_overlapping_repeated_spans_are_counted_once(self):
        phrase = [f'word{i}' for i in range(12)]
        unique = [f'unique{i}' for i in range(70)]
        words = phrase + unique + phrase
        text = ' '.join(words)
        expected = 2 * len(' '.join(phrase)) / len(text)
        self.assertAlmostEqual(_dup_ngram_char_share_from_words(words, 5, len(text)), expected)

    def test_fully_repeated_text_still_has_high_repetition(self):
        words = ['repeated'] * 100
        self.assertAlmostEqual(_dup_ngram_char_share_from_words(words, 5, len(' '.join(words))), 1.)

    def test_worked_math_is_not_rejected_for_reusing_latex_operators(self):
        text = r'''How do you express sin(pi/8) * cos(5 pi/8) without products of trigonometric functions?
$- {\sin}^{2} \left(\frac{\pi}{8}\right)$
$\cos \left(\frac{5 \pi}{8}\right) = \cos \left(\frac{\pi}{8} + \frac{4 \pi}{8}\right) = \cos \left(\frac{\pi}{8} + \frac{\pi}{2}\right) = - \sin \left(\frac{\pi}{8}\right)$
$P = \sin \left(\frac{\pi}{8}\right) . \cos \left(\frac{5 \pi}{8}\right) = - {\sin}^{2} \left(\frac{\pi}{8}\right)$'''
        self.assertEqual(should_keep_document(text, SpakieConfig(), 'finemath'), (True, 'kept'))
        spam = ('This is a repeated advertisement promising easy money to every reader.\n' * 20)
        self.assertFalse(should_keep_document(spam, SpakieConfig(), 'finemath')[0])


def tiny_config():
    return SpakieConfig(vocab_size=64, n_layers=1, n_heads=2,
                       n_kv_heads=2, d_model=16, d_ff=32,
                       swiglu_hidden=32, max_seq_len=16, dropout=0.)


class TorchPrecisionTests(unittest.TestCase):
    def test_head_keeps_float32_under_autocast_and_backpropagates(self):
        model = SpakieGPT(tiny_config()).to(torch.bfloat16)
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits, loss = model(ids, ids)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(model.tok_emb.weight.grad).all())


@unittest.skipIf(_skip_if_no_mlx(), 'MLX/Metal unavailable')
class MLXPrecisionTests(unittest.TestCase):
    def test_head_is_float32_for_bfloat16_model(self):
        import mlx.core as mx
        from model.transformer_mlx import SpakieGPTMLX
        model = SpakieGPTMLX(tiny_config())
        model.set_dtype(mx.bfloat16)
        logits, _, _ = model(mx.array([[1, 2, 3]]))
        self.assertEqual(logits.dtype, mx.float32)

    def test_custom_and_fused_loss_gradients_match_float32_reference(self):
        import mlx.core as mx
        import mlx.nn as nn
        from model.transformer_mlx import _linear_cross_entropy_mean
        from model.fused_ce_mlx import fused_linear_cross_entropy_mean
        # A large common offset plus sub-BF16-resolution differences.
        x = mx.array([[1., .25, .5], [1., .5, .25]], dtype=mx.bfloat16)
        w = mx.array([[-240., i / 16, -i / 32] for i in range(64)], dtype=mx.bfloat16)
        y = mx.array([2, 31])
        def reference(a, b):
            return nn.losses.cross_entropy(a.astype(mx.float32) @ b.astype(mx.float32).T, y, reduction='mean')
        expected, grads = mx.value_and_grad(reference, argnums=(0, 1))(x, w)
        for fn in [_linear_cross_entropy_mean, fused_linear_cross_entropy_mean]:
            actual, got = mx.value_and_grad(lambda a, b: fn(a, b, y), argnums=(0, 1))(x, w)
            self.assertEqual(actual.dtype, mx.float32)
            self.assertAlmostEqual(actual.item(), expected.item(), places=4)
            for a, b in zip(got, grads):
                np.testing.assert_allclose(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)), atol=.01, rtol=.02)


if __name__ == '__main__':
    unittest.main()
