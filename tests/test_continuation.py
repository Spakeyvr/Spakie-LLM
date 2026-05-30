import unittest

from inference.continuation import decode_prefilled_continuation


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def decode(self, ids, skip_special_tokens=True):
        self.calls.append((ids, skip_special_tokens))
        return "|".join(str(token_id) for token_id in ids)


class ContinuationDisplayTests(unittest.TestCase):
    def test_decodes_prompt_plus_response(self):
        tokenizer = FakeTokenizer()

        text = decode_prefilled_continuation(tokenizer, [10, 11], [12, 13])

        self.assertEqual(text, "10|11|12|13")
        self.assertEqual(tokenizer.calls, [([10, 11, 12, 13], True)])

    def test_show_special_tokens_disables_special_token_skipping(self):
        tokenizer = FakeTokenizer()

        decode_prefilled_continuation(
            tokenizer,
            [1],
            [2],
            show_special_tokens=True,
        )

        self.assertEqual(tokenizer.calls, [([1, 2], False)])


if __name__ == "__main__":
    unittest.main()
