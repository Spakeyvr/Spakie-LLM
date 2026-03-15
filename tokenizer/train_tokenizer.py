"""Train a SentencePiece BPE tokenizer and provide a wrapper class."""

import glob
import os
import tempfile

import sentencepiece as spm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "<|json|>"]


def train_tokenizer(config: SpakieConfig | None = None):
    config = config or SpakieConfig()

    md_files = glob.glob(os.path.join(config.raw_data_dir, "**", "*.md"), recursive=True)
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {config.raw_data_dir}")

    # SentencePiece needs a single input file or comma-separated list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        for path in md_files:
            with open(path, "r", encoding="utf-8") as f:
                tmp.write(f.read())
                tmp.write("\n")
        tmp_path = tmp.name

    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=config.tokenizer_prefix,
            vocab_size=config.vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=SPECIAL_TOKENS,
            byte_fallback=True,
            normalization_rule_name="identity",
            num_threads=os.cpu_count(),
        )
    finally:
        os.unlink(tmp_path)

    print(f"Tokenizer saved to {config.tokenizer_prefix}.model")


class SpakieTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    @property
    def user_id(self) -> int:
        return self.piece_to_id("<|user|>")

    @property
    def assistant_id(self) -> int:
        return self.piece_to_id("<|assistant|>")

    @property
    def system_id(self) -> int:
        return self.piece_to_id("<|system|>")

    @property
    def json_id(self) -> int:
        return self.piece_to_id("<|json|>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


if __name__ == "__main__":
    train_tokenizer()
