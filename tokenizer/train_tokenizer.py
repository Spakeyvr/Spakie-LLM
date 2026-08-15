"""Train a cleaned, source-weighted SentencePiece tokenizer and provide a wrapper."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import sentencepiece as spm
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.default import SpakieConfig


SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|system|>", "<|json|>"]
SUPPORTED_EXTENSIONS = (".md", ".txt", ".jsonl")
# Field names tried in order when extracting text from JSONL records.
_JSONL_TEXT_KEYS = ("text", "content", "input", "instruction", "output")
TOKENIZER_MAX_SENTENCE_BYTES = 4096


def _extract_jsonl_text(payload: dict) -> str:
    for key in _JSONL_TEXT_KEYS:
        val = payload.get(key, "")
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _iter_path_texts(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = _extract_jsonl_text(payload)
                        if text:
                            yield text
            else:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    yield text
        except OSError:
            continue


def iter_sentencepiece_chunks(
    text: str, *, max_bytes: int = TOKENIZER_MAX_SENTENCE_BYTES
) -> Iterator[str]:
    """Yield non-empty one-line samples that fit SentencePiece's byte limit."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    # Paragraph boundaries preserve useful local context. Whitespace inside a
    # sample is normalized because SentencePiece consumes one sentence per line.
    for paragraph in re.split(r"\n\s*\n", text):
        words = paragraph.split()
        if not words:
            continue
        current: list[str] = []
        current_bytes = 0
        for word in words:
            encoded = word.encode("utf-8")
            separator = 1 if current else 0
            if current and current_bytes + separator + len(encoded) > max_bytes:
                yield " ".join(current)
                current = []
                current_bytes = 0
                separator = 0
            if len(encoded) > max_bytes:
                if current:
                    yield " ".join(current)
                    current = []
                    current_bytes = 0
                raw = encoded
                while raw:
                    cut = min(max_bytes, len(raw))
                    while cut > 0:
                        try:
                            piece = raw[:cut].decode("utf-8")
                            break
                        except UnicodeDecodeError:
                            cut -= 1
                    if cut == 0:
                        break
                    if piece:
                        yield piece
                    raw = raw[cut:]
                continue
            current.append(word)
            current_bytes += separator + len(encoded)
        if current:
            yield " ".join(current)


def _source_name(root: Path, path: Path) -> str:
    parts = path.relative_to(root).parts
    if len(parts) >= 3 and parts[0] == "large_corpus":
        return parts[1]
    return parts[0] if len(parts) > 1 else "__root__"


def iter_training_texts(
    raw_root: str,
    *,
    source_weights: dict[str, int] | None = None,
    text_transform: Callable[[str, str], str | None] | None = None,
    on_sample: Callable[[str, int], None] | None = None,
):
    """Yield a deterministic weighted sample across corpus sources.

    The tokenizer cap must not mean "the first five million records in lexical
    path order": on the real corpus that excluded every later source. Weighted
    fair scheduling follows the final corpus plan while keeping memory bounded
    by source count and retaining deterministic output.
    """
    root = Path(raw_root)
    grouped: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        grouped.setdefault(_source_name(root, path), []).append(path)

    def source_chunks(source: str) -> Iterator[str]:
        for text in _iter_path_texts(grouped[source]):
            if text_transform is not None:
                text = text_transform(source, text)
                if not text:
                    continue
            yield from iter_sentencepiece_chunks(text)

    weights = {
        source: max(1, int((source_weights or {}).get(source, 1)))
        for source in grouped
    }
    active = {source: source_chunks(source) for source in sorted(grouped)}
    credit = {source: 0.0 for source in active}
    while active:
        total_weight = sum(weights[source] for source in active)
        for source in active:
            credit[source] += weights[source] / total_weight
        source = max(sorted(active), key=lambda name: credit[name])
        try:
            sample = next(active[source])
            if on_sample is not None:
                on_sample(source, len(sample.encode("utf-8")))
            yield sample
            credit[source] -= 1.0
        except StopIteration:
            del active[source]
            del credit[source]


def _clean_tokenizer_texts(config: SpakieConfig) -> Callable[[str, str], str | None]:
    """Build the canonical clean/filter gate for tokenizer samples."""
    # Lazy imports avoid a module cycle: prepare_data imports SpakieTokenizer.
    from runtime.langid import is_probably_english
    from scripts.prepare_data import (
        clean_text,
        language_filter_sample,
        should_keep_document,
    )

    seen: set[bytes] = set()

    def transform(source: str, raw_text: str) -> str | None:
        text = clean_text(raw_text, source)
        keep, _reason = should_keep_document(text, config, source)
        if not keep:
            return None
        language_sample = language_filter_sample(text, config, source)
        if language_sample is not None and not is_probably_english(
            language_sample, config
        ):
            return None
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if digest in seen:
            return None
        seen.add(digest)
        return text

    return transform


def train_tokenizer(config: SpakieConfig | None = None, max_sentences: int = 5_000_000):
    """Train SentencePiece from cleaned text weighted to the corpus plan.

    Args:
        config: SpakieConfig. Uses defaults if None.
        max_sentences: Cap on bounded samples written to the training file. Prevents
                       multi-GB temp files when training on large corpora.
    """
    config = config or SpakieConfig()
    max_sentences = max(1, int(max_sentences))
    count = 0
    bytes_written = 0
    source_counts: Counter[str] = Counter()
    current_source = ""
    output_prefix = Path(config.tokenizer_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="spakie-tokenizer-",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = tmp_file.name
    temporary_prefix = output_prefix.parent / (
        f".{output_prefix.name}.{os.getpid()}.tmp"
    )

    def note_sample(source: str, _sample_bytes: int) -> None:
        nonlocal current_source
        current_source = source
        source_counts[source] += 1

    try:
        source_weights = {
            source: int(plan.get("target_tokens", 0))
            for source, plan in config.corpus_source_plan.items()
            if plan.get("enabled", True) and int(plan.get("target_tokens", 0)) > 0
        }
        with tmp_file as tmp, tqdm(
            total=max_sentences,
            desc="Collecting tokenizer samples",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            for sentence in iter_training_texts(
                config.raw_data_dir,
                source_weights=source_weights,
                text_transform=_clean_tokenizer_texts(config),
                on_sample=note_sample,
            ):
                tmp.write(sentence)
                tmp.write("\n")
                count += 1
                bytes_written += len(sentence.encode("utf-8")) + 1
                progress.update(1)
                if count % 1_000 == 0:
                    progress.set_postfix_str(
                        f"{bytes_written / (1024 ** 3):.2f} GiB, {current_source}",
                        refresh=False,
                    )
                if count >= max_sentences:
                    break

        if count == 0:
            raise FileNotFoundError(f"No training texts found in {config.raw_data_dir!r}")

        mix = ", ".join(
            f"{source}={amount / count:.1%}"
            for source, amount in source_counts.most_common()
        )
        print(f"Collected {count:,} samples ({bytes_written / (1024 ** 3):.2f} GiB): {mix}")
        print(f"Training {config.vocab_size:,}-piece SentencePiece tokenizer …", flush=True)
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=str(temporary_prefix),
            vocab_size=config.vocab_size,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=SPECIAL_TOKENS,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,   # no leading-space artifact on first token
            split_digits=True,        # each digit is its own token (better for arithmetic)
            character_coverage=0.9999,
            max_sentence_length=TOKENIZER_MAX_SENTENCE_BYTES,
            num_threads=os.cpu_count(),
        )
        os.replace(
            str(temporary_prefix) + ".vocab",
            str(output_prefix) + ".vocab",
        )
        os.replace(
            str(temporary_prefix) + ".model",
            str(output_prefix) + ".model",
        )
    finally:
        for path in (
            Path(tmp_path),
            Path(str(temporary_prefix) + ".model"),
            Path(str(temporary_prefix) + ".vocab"),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    print(f"Saved: {config.tokenizer_prefix}.model  ({config.vocab_size:,} vocab tokens)")


class SpakieTokenizer:
    """SentencePiece BPE tokenizer with chat-template and batch support."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path!r}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Cache all special-token IDs once so every property access is O(1).
        self._pad_id       = self.sp.pad_id()
        self._unk_id       = self.sp.unk_id()
        self._bos_id       = self.sp.bos_id()
        self._eos_id       = self.sp.eos_id()
        self._user_id      = self.sp.piece_to_id("<|user|>")
        self._assistant_id = self.sp.piece_to_id("<|assistant|>")
        self._system_id    = self.sp.piece_to_id("<|system|>")
        self._json_id      = self.sp.piece_to_id("<|json|>")

        # Set of IDs that should never appear in decoded output text.
        self._control_ids = frozenset({
            self._pad_id, self._bos_id, self._eos_id,
            self._user_id, self._assistant_id, self._system_id, self._json_id,
        })

    # ── Python protocols ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def __repr__(self) -> str:
        return f"SpakieTokenizer(vocab_size={len(self)})"

    def __contains__(self, piece: str) -> bool:
        """Return True if piece is a known (non-UNK) vocabulary token."""
        return self.sp.piece_to_id(piece) != self._unk_id

    # ── ID accessors (all O(1)) ────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self)

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def unk_id(self) -> int:
        return self._unk_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def user_id(self) -> int:
        return self._user_id

    @property
    def assistant_id(self) -> int:
        return self._assistant_id

    @property
    def system_id(self) -> int:
        return self._system_id

    @property
    def json_id(self) -> int:
        return self._json_id

    # ── Vocabulary helpers ─────────────────────────────────────────────────

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def id_to_piece(self, token_id: int) -> str:
        return self.sp.id_to_piece(token_id)

    # ── Encoding ───────────────────────────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode a single string to a list of token IDs."""
        ids: list[int] = self.sp.encode(text)
        if add_bos:
            ids = [self._bos_id] + ids
        if add_eos:
            ids = ids + [self._eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_bos: bool = False,
                     add_eos: bool = False, num_threads: int = -1) -> list[list[int]]:
        """Encode a list of strings in parallel. Returns a list of ID lists.

        Args:
            texts: Input strings.
            add_bos: Prepend BOS to every sequence.
            add_eos: Append EOS to every sequence.
            num_threads: Worker threads (-1 = auto).
        """
        results: list[list[int]] = self.sp.encode(texts, num_threads=num_threads)
        if add_bos or add_eos:
            for i, ids in enumerate(results):
                if add_bos:
                    ids = [self._bos_id] + ids
                if add_eos:
                    ids = ids + [self._eos_id]
                results[i] = ids
        return results

    def encode_as_pieces(self, text: str) -> list[str]:
        """Return the string tokens for text (useful for debugging coverage)."""
        return self.sp.encode_as_pieces(text)

    # ── Decoding ───────────────────────────────────────────────────────────

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to a string.

        Args:
            ids: Token ID sequence.
            skip_special_tokens: Strip pad/BOS/EOS/role tokens before decoding
                                 so they never appear literally in the output.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._control_ids]
        return self.sp.decode(ids)

    def decode_batch(self, batch: list[list[int]],
                     skip_special_tokens: bool = True) -> list[str]:
        """Decode a batch of ID lists to strings."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    # ── Chat template ──────────────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict],
        system_msg: str = "",
        add_assistant_prompt: bool = True,
    ) -> list[int]:
        """Render a conversation to a flat token ID sequence.

        Format (each turn):
            [role_id] <text tokens> [eos_id]

        Args:
            messages: List of {"role": str, "content": str} dicts.
                      Recognised roles: "system", "user", "assistant".
            system_msg: Optional system message prepended before messages.
            add_assistant_prompt: Append assistant_id to cue generation.
        """
        ids: list[int] = []

        if system_msg:
            ids += [self._system_id] + self.encode(system_msg) + [self._eos_id]

        _role_map = {
            "system":    self._system_id,
            "user":      self._user_id,
            "assistant": self._assistant_id,
        }

        for msg in messages:
            role_token = _role_map.get(msg.get("role", ""))
            if role_token is None:
                continue
            ids += [role_token] + self.encode(msg.get("content", "")) + [self._eos_id]

        if add_assistant_prompt:
            ids.append(self._assistant_id)

        return ids

    # ── Metrics ────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens text encodes to."""
        return len(self.sp.encode(text))

    def compression_ratio(self, text: str) -> float:
        """Characters per token — a proxy for tokeniser efficiency."""
        n = self.count_tokens(text)
        return len(text) / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a cleaned, corpus-weighted SentencePiece tokenizer"
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5_000_000,
        help="Maximum cleaned SentencePiece samples (default: 5,000,000)",
    )
    args = parser.parse_args()
    train_tokenizer(max_sentences=max(1, args.max_sentences))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTokenizer training interrupted; temporary input was removed.")
        raise SystemExit(130)
