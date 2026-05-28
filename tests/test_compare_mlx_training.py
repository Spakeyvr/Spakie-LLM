import unittest

from scripts.compare_mlx_training import _parse_result


class CompareMLXTrainingTests(unittest.TestCase):
    def test_parse_result_extracts_speed_and_thermal_fields(self):
        output = "\n".join(
            [
                "Task: pretrain",
                "Avg step: 1234.50 ms",
                "Iterations/s: 0.8100",
                "Tokens/step: 65536",
                "Throughput: 53084 tok/s",
                "Supervised tokens/step: 49152.0",
                "Supervised throughput: 39813 tok/s",
                "Thermal before: Note: cool",
                "Thermal after: Note: still cool",
            ]
        )

        result = _parse_result("candidate", 2, output)

        self.assertEqual(result.label, "candidate")
        self.assertEqual(result.round_index, 2)
        self.assertEqual(result.avg_step_ms, 1234.50)
        self.assertEqual(result.iter_per_sec, 0.8100)
        self.assertEqual(result.tokens_per_step, 65536)
        self.assertEqual(result.throughput, 53084)
        self.assertEqual(result.supervised_tokens_per_step, 49152.0)
        self.assertEqual(result.supervised_throughput, 39813)
        self.assertEqual(result.thermal_before, "Note: cool")
        self.assertEqual(result.thermal_after, "Note: still cool")


if __name__ == "__main__":
    unittest.main()
