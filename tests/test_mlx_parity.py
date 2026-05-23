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
            overrides[f"{src_prefix}.mlp.fc1.weight"] = mx.array(state[f"{src_prefix}.mlp.fc1.weight"])
            overrides[f"{src_prefix}.mlp.fc2.weight"] = mx.array(state[f"{src_prefix}.mlp.fc2.weight"])

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


if __name__ == "__main__":
    unittest.main()
