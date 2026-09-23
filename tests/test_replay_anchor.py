import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'MLX unavailable')
class ReplayAnchorTests(unittest.TestCase):
    def test_kl_direction_mask_normalization_and_teacher_gradient(self):
        import mlx.core as mx
        from scripts._replay_anchor import replay_kl
        student = mx.log(mx.array([[[.4,.6]], [[.1,.9]]]))
        teacher = mx.log(mx.array([[[.8,.2]], [[.7,.3]]]))
        mask = mx.array([True,False])
        loss = replay_kl(student, teacher, mask)
        expected = (.8*np.log(.8/.4)+.2*np.log(.2/.6))/2
        self.assertAlmostEqual(float(loss.item()), expected, places=6)
        grad = mx.grad(lambda s: replay_kl(s, teacher, mask))(student)
        np.testing.assert_allclose(np.asarray(grad), [[[-.2,.2]], [[0.,0.]]], atol=1e-7)
        teacher_grad = mx.grad(lambda t: replay_kl(student, t, mask))(teacher)
        np.testing.assert_array_equal(np.asarray(teacher_grad), np.zeros((2,1,2)))
        self.assertAlmostEqual(float(replay_kl(student+200, teacher-200, mask).item()), expected, places=5)
        self.assertEqual(float(replay_kl(student, teacher, mx.array([False,False])).item()), 0.)
        self.assertAlmostEqual(float(replay_kl(student, student, mask).item()), 0., places=7)
        stationary_grad = mx.grad(lambda s: replay_kl(s, student, mask))(student)
        np.testing.assert_array_equal(np.asarray(stationary_grad),np.zeros((2,1,2)))

    def test_zero_anchor_matches_existing_loss_and_model_gradients(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from configs.default import SpakieConfig
        from model.transformer_mlx import SpakieGPTMLX
        from scripts._replay_anchor import build_anchored_step
        from training.pretrain_mlx import _build_microbatch_step
        config = SpakieConfig(vocab_size=128, n_layers=2, n_heads=4, n_kv_heads=2,
                              d_model=32, d_ff=64, swiglu_hidden=32, mlp_type='swiglu',
                              max_seq_len=16, dropout=0., bias=False, activation_checkpointing=False)
        mx.random.seed(42)
        model = SpakieGPTMLX(config)
        teacher = SpakieGPTMLX(config)
        teacher.load_weights(tree_flatten(model.parameters()))
        teacher.eval()
        before = {k:np.asarray(v).copy() for k,v in tree_flatten(teacher.parameters())}
        x = mx.array([[1,7,3,9],[4,3,2,8]])
        y = mx.array([[7,3,9,2],[3,2,8,1]])
        reference, reference_grads = _build_microbatch_step(model,1.,compile_step=False,ignore_index=None)(x,y)
        (loss, ce, kl), grads = build_anchored_step(model,teacher,0.)(x,y,mx.array([True,False]))
        mx.eval(reference,loss,ce,kl,grads,reference_grads)
        self.assertAlmostEqual(float(reference.item()),float(loss.item()),places=5)
        self.assertAlmostEqual(float(kl.item()),0.,places=6)
        for (name, actual),(other_name,expected) in zip(tree_flatten(grads),tree_flatten(reference_grads)):
            self.assertEqual(name,other_name)
            np.testing.assert_allclose(np.asarray(actual),np.asarray(expected),rtol=2e-4,atol=2e-6)
        for name,value in tree_flatten(teacher.parameters()):
            np.testing.assert_array_equal(np.asarray(value),before[name])
        (stationary_loss, _, _), stationary_grads = build_anchored_step(
            model,teacher,1.,replace_replay_ce=True)(x,y,mx.array([True,True]))
        self.assertAlmostEqual(float(stationary_loss.item()),0.,places=6)
        for _,value in tree_flatten(stationary_grads):
            np.testing.assert_array_equal(np.asarray(value),np.zeros(value.shape))


if __name__ == '__main__':
    unittest.main()
