import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.dataset_mlx import ChatSFTDatasetMLX, PackedChatSFTDatasetMLX


class FakeTokenizer:
    system_id = 1
    user_id = 2
    assistant_id = 3
    eos_id = 4
    pad_id = 0

    def encode(self, text):
        return [10 + (ord(ch) % 50) for ch in text]


class PackedSFTDatasetMLXTests(unittest.TestCase):
    def test_packed_dataset_preserves_supervised_labels(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "bc"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "d"},
                    {"role": "assistant", "content": "ef"},
                ]
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            plain = ChatSFTDatasetMLX(str(path), FakeTokenizer(), max_seq_len=8)
            packed = PackedChatSFTDatasetMLX(str(path), FakeTokenizer(), max_seq_len=8)

        plain_supervised = sum(int((plain[i][1] != -100).sum()) for i in range(len(plain)))
        packed_supervised = sum(int((packed[i][1] != -100).sum()) for i in range(len(packed)))
        self.assertEqual(plain_supervised, packed_supervised)
        self.assertGreaterEqual(len(plain), len(packed))
        for i in range(len(packed)):
            x, y = packed[i]
            self.assertEqual(x.dtype, np.int32)
            self.assertEqual(y.dtype, np.int32)
            self.assertEqual(x.shape, (8,))
            self.assertEqual(y.shape, (8,))


if __name__ == "__main__":
    unittest.main()
