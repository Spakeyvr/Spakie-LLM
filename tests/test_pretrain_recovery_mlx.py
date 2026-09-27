"""Real production-loop SIGINT/resume equivalence on a tiny BASE model."""
import os
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('mlx') is not None, 'MLX unavailable')
class PretrainRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe = subprocess.run([sys.executable, '-c',
                'import mlx.core as mx; assert mx.metal.is_available(); mx.eval(mx.array([1.]))'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            if probe.returncode:
                raise unittest.SkipTest('MLX/Metal unavailable')
        except (OSError, subprocess.TimeoutExpired):
            raise unittest.SkipTest('MLX/Metal unavailable')

    def test_interrupt_resume_restores_exact_state_and_sampler_with_stable_loss(self):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from configs.default import get_preset_config
        from model.transformer_mlx import SpakieGPTMLX
        from runtime.mlx_backend import resolve_mlx_runtime
        from training.dataset_mlx import PretrainDatasetMLX, ResumableBatchSamplerMLX
        from training.optimizers_mlx import configure_mlx_optimizer
        from training.pretrain_mlx import pretrain_mlx, load_training_checkpoint_mlx, save_training_checkpoint_mlx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # A self-contained numerical fixture, not a capability experiment.
            tokens = np.random.default_rng(41).integers(8, 128, size=4097, dtype=np.uint16)
            np.save(root/'tokens.npy', tokens)
            dataset = PretrainDatasetMLX(str(root/'tokens.npy'), seq_len=32)
            cfg = get_preset_config('92m')
            cfg.vocab_size, cfg.n_layers, cfg.d_model = 128, 2, 64
            cfg.n_heads, cfg.n_kv_heads = 4, 2
            cfg.d_ff = cfg.swiglu_hidden = 128
            cfg.max_seq_len, cfg.dropout = 32, 0.
            cfg.pretrain_batch_size, cfg.pretrain_grad_accum_steps = 2, 3
            cfg.pretrain_target_tokens = 4 * cfg.pretrain_tokens_per_step()
            cfg.refresh_derived_fields()
            cfg.pretrain_warmup_steps = 1
            cfg.pretrain_eval_interval = 100
            cfg.pretrain_checkpoint_interval = 0
            cfg.tokenizer_prefix = str(root/'unused-tokenizer')
            cfg.processed_data_dir = str(root/'unused-data')
            runtime = resolve_mlx_runtime('bf16')

            def host(value):
                if isinstance(value, mx.array) and value.dtype == mx.bfloat16:
                    value = value.astype(mx.float32)
                return np.asarray(value)

            def assert_trees_equal(left, right):
                left, right = dict(tree_flatten(left)), dict(tree_flatten(right))
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    np.testing.assert_array_equal(host(left[key]), host(right[key]), err_msg=key)

            def checked_save(path, **kw):
                # Check actual bytes restored from disk against the live state
                # at the committed optimizer boundary, before more GPU work.
                save_training_checkpoint_mlx(path, **kw)
                loaded = load_training_checkpoint_mlx(path)
                assert_trees_equal(kw['model'].parameters(), loaded['model'])
                assert_trees_equal(kw['optimizer'].state_trees(), loaded['optimizer'])
                self.assertEqual(loaded['meta']['step'], kw['global_step'])

            def run(name, resume=None, interrupt=False):
                cfg.checkpoint_dir = str(root/name)
                mx.random.seed(19)
                model = SpakieGPTMLX(cfg)
                sampler = (ResumableBatchSamplerMLX.from_state_dict(resume['meta']['sampler'])
                           if resume else ResumableBatchSamplerMLX(len(dataset), 2, seed=17))
                def build_optimizer(*args, **kwargs):
                    optimizer = configure_mlx_optimizer(*args, **kwargs)
                    restore = optimizer.load_state_trees
                    def checked_restore(state):
                        restore(state)
                        assert_trees_equal(state, optimizer.state_trees())
                    optimizer.load_state_trees = checked_restore
                    update = optimizer.update
                    count = 0
                    def wrapped_update(*a, **kw):
                        nonlocal count
                        update(*a, **kw)
                        count += 1
                        if interrupt and count == 2:
                            os.kill(os.getpid(), signal.SIGINT)
                    optimizer.update = wrapped_update
                    return optimizer
                with patch('training.pretrain_mlx.configure_mlx_optimizer', side_effect=build_optimizer), \
                     patch('training.pretrain_mlx.save_training_checkpoint_mlx', side_effect=checked_save):
                    pretrain_mlx(model, dataset, dataset, sampler, cfg, runtime, resume_state=resume,
                                 use_compile=True, use_prefetch=True)
                suffix = 'interrupt' if interrupt else 'final'
                return load_training_checkpoint_mlx(str(root/name/f'pretrain_{suffix}.safetensors'))

            uninterrupted = run('continuous')
            interrupted = run('interrupted', interrupt=True)
            self.assertEqual(interrupted['meta']['step'], 2)
            resumed = run('resumed', resume=interrupted)
            self.assertEqual(resumed['meta']['step'], 4)
            self.assertEqual(resumed['meta']['tokens_processed'], 4*cfg.pretrain_tokens_per_step())
            # Metal backward reductions can differ slightly even across fresh
            # identical runs. Check inference equivalence at a stated tolerance;
            # disk serialization itself was checked bit-for-bit above.
            losses = []
            x, y = dataset.get_batch([0,1])
            for checkpoint in (uninterrupted, resumed):
                model = SpakieGPTMLX(cfg)
                model.load_weights(tree_flatten(checkpoint['model']), strict=True)
                model.eval()
                _, loss, _ = model(mx.array(x), mx.array(y), ignore_index=None)
                losses.append(float(loss.item()))
            self.assertLess(abs(losses[0] - losses[1]), .002)
            left, right = uninterrupted['meta']['sampler'], resumed['meta']['sampler']
            self.assertEqual(left['position'], right['position'])
            np.testing.assert_array_equal(left['indices'], right['indices'])
            np.testing.assert_array_equal(next(iter(ResumableBatchSamplerMLX.from_state_dict(left))),
                                          next(iter(ResumableBatchSamplerMLX.from_state_dict(right))))


if __name__ == '__main__':
    unittest.main()
