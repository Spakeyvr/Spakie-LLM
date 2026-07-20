import unittest

import mlx.core as mx

from scripts.interpolate_checkpoints_mlx import interpolate_arrays


class InterpolateCheckpointsMLXTests(unittest.TestCase):
    def setUp(self):
        try:
            probe = mx.array([0.0])
            mx.eval(probe)
        except RuntimeError as exc:
            self.skipTest(f"MLX/Metal unavailable: {exc}")

    def test_endpoints_and_midpoint(self):
        base = {"model.w": mx.array([0.0, 2.0], dtype=mx.float32)}
        target = {"model.w": mx.array([2.0, 6.0], dtype=mx.float32)}
        self.assertEqual(interpolate_arrays(base, target, 0.0)["model.w"].tolist(), [0.0, 2.0])
        self.assertEqual(interpolate_arrays(base, target, 0.5)["model.w"].tolist(), [1.0, 4.0])
        self.assertEqual(interpolate_arrays(base, target, 1.0)["model.w"].tolist(), [2.0, 6.0])

    def test_rejects_invalid_alpha_and_key_mismatch(self):
        one = {"a": mx.array([1.0])}
        with self.assertRaises(ValueError):
            interpolate_arrays(one, one, 1.1)
        with self.assertRaises(ValueError):
            interpolate_arrays(one, {"b": mx.array([1.0])}, 0.5)

    def test_selected_keys_leave_other_tensors_at_base(self):
        base = {"a": mx.array([0.0]), "b": mx.array([10.0])}
        target = {"a": mx.array([4.0]), "b": mx.array([20.0])}
        result = interpolate_arrays(base, target, 1.0, {"b"})
        self.assertEqual(result["a"].tolist(), [0.0])
        self.assertEqual(result["b"].tolist(), [20.0])


if __name__ == "__main__":
    unittest.main()
