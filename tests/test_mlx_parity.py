"""Forward-pass parity test between the PyTorch and MLX transformer implementations.

Seeds both models with the same weights (copied from the PyTorch state dict) and
checks that the produced logits agree within a small tolerance. The point is to
catch algebra/index/mask bugs when porting — not to regress on bitwise identity.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.default import SpakieConfig
from model.transformer import SpakieGPT


def _skip_if_no_mlx():
    try:
        import mlx.core  # noqa: F401

        return False
    except ImportError:
        return True


@unittest.skipIf(_skip_if_no_mlx(), "mlx not installed")
class TorchMLXForwardParityTests(unittest.TestCase):
    @staticmethod
    def _tiny_config() -> SpakieConfig:
        return SpakieConfig(
            vocab_size=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_model=32,
            d_ff=64,
            mlp_type="swiglu",
            swiglu_hidden=32,
            max_seq_len=16,
            dropout=0.0,
            bias=False,
        )

    def _copy_torch_weights_into_mlx(self, torch_model: SpakieGPT, mlx_model) -> None:
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        state = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}
        # tok_emb and pos_emb names match directly.
        overrides: dict[str, mx.array] = {}
        overrides["tok_emb.weight"] = mx.array(state["tok_emb.weight"])
        overrides["pos_emb.weight"] = mx.array(state["pos_emb.weight"])
        overrides["ln_f.weight"] = mx.array(state["ln_f.weight"])
        if "ln_f.bias" in state:
            overrides["ln_f.bias"] = mx.array(state["ln_f.bias"])

        n_layers = torch_model.config.n_layers
        for i in range(n_layers):
            src_prefix = f"blocks.{i}"
            overrides[f"{src_prefix}.ln1.weight"] = mx.array(state[f"{src_prefix}.ln1.weight"])
            overrides[f"{src_prefix}.ln2.weight"] = mx.array(state[f"{src_prefix}.ln2.weight"])
            if f"{src_prefix}.ln1.bias" in state:
                overrides[f"{src_prefix}.ln1.bias"] = mx.array(state[f"{src_prefix}.ln1.bias"])
            if f"{src_prefix}.ln2.bias" in state:
                overrides[f"{src_prefix}.ln2.bias"] = mx.array(state[f"{src_prefix}.ln2.bias"])
            qkv_key = f"{src_prefix}.attn.qkv.weight"
            if qkv_key in state:
                overrides[qkv_key] = mx.array(state[qkv_key])
            else:
                overrides[f"{src_prefix}.attn.q_proj.weight"] = mx.array(
                    state[f"{src_prefix}.attn.q_proj.weight"]
                )
                overrides[f"{src_prefix}.attn.kv_proj.weight"] = mx.array(
                    state[f"{src_prefix}.attn.kv_proj.weight"]
                )
            overrides[f"{src_prefix}.attn.out_proj.weight"] = mx.array(state[f"{src_prefix}.attn.out_proj.weight"])
            q_norm_key = f"{src_prefix}.attn.q_norm.weight"
            if q_norm_key in state:
                overrides[q_norm_key] = mx.array(state[q_norm_key])
                overrides[f"{src_prefix}.attn.k_norm.weight"] = mx.array(
                    state[f"{src_prefix}.attn.k_norm.weight"]
                )
            fc1_key = f"{src_prefix}.mlp.fc1.weight"
            if fc1_key in state:
                overrides[fc1_key] = mx.array(state[fc1_key])
                overrides[f"{src_prefix}.mlp.fc2.weight"] = mx.array(
                    state[f"{src_prefix}.mlp.fc2.weight"]
                )
            else:
                overrides[f"{src_prefix}.mlp.gate_up.weight"] = mx.array(
                    state[f"{src_prefix}.mlp.gate_up.weight"]
                )
                overrides[f"{src_prefix}.mlp.down.weight"] = mx.array(
                    state[f"{src_prefix}.mlp.down.weight"]
                )

        mlx_model.update(tree_unflatten(list(overrides.items())))

    def test_forward_logits_match_within_tolerance(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()

        torch.manual_seed(0)
        torch_model = SpakieGPT(config)
        torch_model.eval()

        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()
        self._copy_torch_weights_into_mlx(torch_model, mlx_model)

        ids = np.array([[3, 7, 11, 1, 5, 9, 2, 4]], dtype=np.int32)

        with torch.no_grad():
            torch_logits, _ = torch_model(torch.from_numpy(ids.astype(np.int64)))
        torch_logits_np = torch_logits.detach().cpu().numpy()

        mlx_logits, _, _ = mlx_model(mx.array(ids))
        mx.eval(mlx_logits)
        mlx_logits_np = np.asarray(mlx_logits.astype(mx.float32))

        self.assertEqual(torch_logits_np.shape, mlx_logits_np.shape)
        max_abs = np.max(np.abs(torch_logits_np - mlx_logits_np))
        self.assertLess(max_abs, 1e-3, f"max abs logit diff = {max_abs}")

    def test_forward_logits_match_with_qk_norm(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        config.qk_norm = True

        torch.manual_seed(1)
        torch_model = SpakieGPT(config)
        # Randomize the QK-norm gains so the test catches axis/order bugs that a
        # ones-initialized norm would hide.
        with torch.no_grad():
            for block in torch_model.blocks:
                block.attn.q_norm.weight.normal_(mean=1.0, std=0.1)
                block.attn.k_norm.weight.normal_(mean=1.0, std=0.1)
        torch_model.eval()

        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()
        self._copy_torch_weights_into_mlx(torch_model, mlx_model)

        ids = np.array([[3, 7, 11, 1, 5, 9, 2, 4]], dtype=np.int32)
        with torch.no_grad():
            torch_logits, _ = torch_model(torch.from_numpy(ids.astype(np.int64)))
        torch_logits_np = torch_logits.detach().cpu().numpy()

        mlx_logits, _, _ = mlx_model(mx.array(ids))
        mx.eval(mlx_logits)
        mlx_logits_np = np.asarray(mlx_logits.astype(mx.float32))

        self.assertEqual(torch_logits_np.shape, mlx_logits_np.shape)
        max_abs = np.max(np.abs(torch_logits_np - mlx_logits_np))
        self.assertLess(max_abs, 1e-3, f"max abs logit diff = {max_abs}")

    def test_return_cache_disabled_returns_no_cache_payload(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()

        ids = np.array([[3, 7, 11]], dtype=np.int32)
        logits, loss, cache = mlx_model(mx.array(ids), return_cache=False)
        mx.eval(logits)

        self.assertEqual(tuple(logits.shape), (1, ids.shape[1], config.vocab_size))
        self.assertIsNone(loss)
        self.assertIsNone(cache)

    def test_checkpoint_loader_rejects_actual_partial_model_tree(self):
        from mlx.utils import tree_flatten

        from model.transformer_mlx import SpakieGPTMLX
        from runtime.checkpoint_io import load_mlx_model_weights_strict

        source = SpakieGPTMLX(self._tiny_config())
        source_flat = dict(tree_flatten(source.parameters()))
        first_key = next(iter(source_flat))

        target = SpakieGPTMLX(self._tiny_config())
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            load_mlx_model_weights_strict(
                target,
                {f"model.{first_key}": source_flat[first_key]},
                path="partial.safetensors",
            )

        # The same boundary accepts a complete, shape-compatible checkpoint.
        load_mlx_model_weights_strict(
            target,
            {f"model.{key}": value for key, value in source_flat.items()},
            path="complete.safetensors",
        )

    def test_return_cache_enabled_tracks_warmup_and_decode_cache(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()

        prompt = np.array([[3, 7, 11]], dtype=np.int32)
        logits, _, cache = mlx_model(mx.array(prompt), return_cache=True)

        self.assertIsNotNone(cache)
        self.assertEqual(len(cache), config.n_layers)
        cache_tensors = [logits]
        n_kv_heads = config.n_kv_heads or config.n_heads
        for layer_cache in cache:
            self.assertIsNotNone(layer_cache)
            k, v = layer_cache
            cache_tensors.extend((k, v))
            self.assertEqual(tuple(k.shape), (1, n_kv_heads, prompt.shape[1], config.d_model // config.n_heads))
            self.assertEqual(tuple(v.shape), (1, n_kv_heads, prompt.shape[1], config.d_model // config.n_heads))
        mx.eval(*cache_tensors)

        next_token = np.array([[5]], dtype=np.int32)
        next_logits, _, next_cache = mlx_model(
            mx.array(next_token),
            cache=cache,
            cache_offset=prompt.shape[1],
            return_cache=True,
        )

        self.assertIsNotNone(next_cache)
        self.assertEqual(len(next_cache), config.n_layers)
        next_cache_tensors = [next_logits]
        for layer_cache in next_cache:
            self.assertIsNotNone(layer_cache)
            k, v = layer_cache
            next_cache_tensors.extend((k, v))
            self.assertEqual(tuple(k.shape), (1, n_kv_heads, prompt.shape[1] + 1, config.d_model // config.n_heads))
            self.assertEqual(tuple(v.shape), (1, n_kv_heads, prompt.shape[1] + 1, config.d_model // config.n_heads))
        mx.eval(*next_cache_tensors)

    def test_parallel_residual_cache_warmup_matches_full_forward(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        config.residual_type = "parallel"
        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()

        prompt = np.array([[3, 7, 11, 5]], dtype=np.int32)
        full_logits, _, _ = mlx_model(mx.array(prompt))
        cached_logits, _, cache = mlx_model(mx.array(prompt), return_cache=True)
        mx.eval(full_logits, cached_logits, *[tensor for pair in cache for tensor in pair])

        np.testing.assert_allclose(
            np.asarray(cached_logits.astype(mx.float32)),
            np.asarray(full_logits.astype(mx.float32)),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_parallel_residual_cached_decode_matches_full_forward(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        config.residual_type = "parallel"
        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()

        prompt = np.array([[3, 7, 11, 5]], dtype=np.int32)
        next_token = np.array([[9]], dtype=np.int32)
        combined = np.concatenate([prompt, next_token], axis=1)
        full_logits, _, _ = mlx_model(mx.array(combined))
        _, _, cache = mlx_model(mx.array(prompt), return_cache=True)
        cached_logits, _, next_cache = mlx_model(
            mx.array(next_token),
            cache=cache,
            cache_offset=prompt.shape[1],
            return_cache=True,
        )
        mx.eval(
            full_logits,
            cached_logits,
            *[tensor for pair in next_cache for tensor in pair],
        )

        # Full-sequence and one-token cached SDPA use different kernel shapes,
        # so small floating-point drift is expected. The original topology bug
        # produced multi-unit logit errors; keep a tight absolute regression
        # bound without demanding bitwise-identical kernels.
        cached_np = np.asarray(cached_logits[:, -1, :].astype(mx.float32))
        full_np = np.asarray(full_logits[:, -1, :].astype(mx.float32))
        self.assertLess(float(np.max(np.abs(cached_np - full_np))), 1e-3)

    def test_packed_segments_match_separate_mlx_logits(self):
        import mlx.core as mx

        from model.transformer_mlx import SpakieGPTMLX

        config = self._tiny_config()
        mlx_model = SpakieGPTMLX(config)
        mlx_model.eval()

        first = np.array([[3, 7, 11, 2]], dtype=np.int32)
        second = np.array([[5, 9, 13]], dtype=np.int32)
        packed = np.array([[3, 7, 11, 2, 5, 9, 13]], dtype=np.int32)
        segments = np.array([[0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)
        positions = np.array([[0, 1, 2, 3, 0, 1, 2]], dtype=np.int32)

        first_logits, _, _ = mlx_model(mx.array(first))
        second_logits, _, _ = mlx_model(mx.array(second))
        packed_logits, _, _ = mlx_model(
            mx.array(packed),
            segment_ids=mx.array(segments),
            position_ids=mx.array(positions),
        )
        mx.eval(first_logits, second_logits, packed_logits)

        first_np = np.asarray(first_logits.astype(mx.float32))
        second_np = np.asarray(second_logits.astype(mx.float32))
        packed_np = np.asarray(packed_logits.astype(mx.float32))
        self.assertLess(np.max(np.abs(first_np[0] - packed_np[0, :4])), 1e-5)
        self.assertLess(np.max(np.abs(second_np[0] - packed_np[0, 4:7])), 1e-5)


if __name__ == "__main__":
    unittest.main()
