"""NumPy/MLX-native datasets and resumable batch iterator for the MLX backend."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.train_tokenizer import SpakieTokenizer
from training.sft_tokenization import (
    EncodedSFTExample,
    encode_sft_example,
    pad_sft_example,
    prepare_sft_examples,
)


class PretrainDatasetMLX:
    """Memory-mapped numpy dataset for pretraining (returns numpy arrays).

    Sequences are non-overlapping `seq_len`-token windows packed back-to-back,
    matching standard LM pretraining (GPT-2/Llama style). The previous design
    treated every token offset as a separate sequence, which produced ~N
    indices for an N-token corpus and overflowed MLX's int32 dim cap on
    multi-billion-token corpora — preventing exact sampler resume.
    """

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.seq_len
        chunk = np.asarray(self.data[start : start + self.seq_len + 1], dtype=np.int32)
        return chunk[:-1], chunk[1:]

    def get_batch(self, indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build contiguous input/target arrays without per-item temporaries."""
        batch_size = len(indices)
        x = np.empty((batch_size, self.seq_len), dtype=np.int32)
        y = np.empty((batch_size, self.seq_len), dtype=np.int32)
        for row, idx in enumerate(indices):
            start = int(idx) * self.seq_len
            x[row] = self.data[start : start + self.seq_len]
            y[row] = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


class ChatSFTDatasetMLX:
    """JSONL chat dataset with loss masking on non-assistant turns (numpy arrays)."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: SpakieTokenizer,
        max_seq_len: int,
        *,
        pretokenize: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.examples, self._encoded, self.validation_stats = prepare_sft_examples(
            self.examples,
            self.tokenizer,
            self.max_seq_len,
            keep_cache=pretokenize,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        encoded = (
            getattr(self, "_encoded", None)[idx]
            if getattr(self, "_encoded", None) is not None
            else encode_sft_example(self.examples[idx], self.tokenizer, self.max_seq_len)
        )
        return pad_sft_example(
            encoded,
            max_seq_len=self.max_seq_len,
            pad_id=self.tokenizer.pad_id,
        )

    def get_batch(self, indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _stack_items(self, indices)


class PretokenizedChatSFTDatasetMLX:
    """In-memory exact cache for SFT `(x, y)` arrays.

    `ChatSFTDatasetMLX.__getitem__` tokenizes JSON messages every time an item
    is fetched. This wrapper preserves the dataset contract and example order,
    but pays that tokenization cost once up front so the training loop only
    stacks NumPy arrays.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = getattr(dataset, "tokenizer", None)
        self.max_seq_len = getattr(dataset, "max_seq_len", None)
        raw_examples = []
        encoded_items: list[EncodedSFTExample] = []
        for idx in range(len(dataset)):
            raw = _raw_example_at(dataset, idx)
            tokenizer = getattr(dataset, "tokenizer", None)
            max_seq_len = getattr(dataset, "max_seq_len", None)
            if raw is not None and tokenizer is not None and max_seq_len is not None:
                encoded_items.append(encode_sft_example(raw, tokenizer, max_seq_len))
            else:
                item = dataset[idx]
                if not isinstance(item, tuple) or len(item) < 2:
                    raise ValueError("pretokenized SFT dataset expects (x, y) items")
                x, y = item[:2]
                keep = (np.asarray(x) != getattr(self.tokenizer, "pad_id", 0)) | (
                    np.asarray(y) != -100
                )
                length = int(np.nonzero(keep)[0].max()) + 1 if keep.any() else 1
                encoded_items.append(EncodedSFTExample(
                    np.asarray(x[:length], dtype=np.int32),
                    np.asarray(y[:length], dtype=np.int32),
                ))
            if raw is not None:
                raw_examples.append(raw)
        self._encoded = encoded_items
        if len(raw_examples) == len(encoded_items):
            self.examples = raw_examples

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return pad_sft_example(
            self._encoded[idx],
            max_seq_len=self.max_seq_len,
            pad_id=self.tokenizer.pad_id,
        )

    def get_batch(self, indices: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _stack_items(self, indices)

    def sequence_lengths(self) -> np.ndarray:
        return np.asarray([item.length for item in self._encoded], dtype=np.int32)


def train_val_split_mlx(dataset, val_fraction: float = 0.05, seed: int = 42):
    """Deterministic grouped split. Returns (train_indices, val_indices).

    Synthetic SFT files often contain repeated prompts. Splitting by row leaks
    those duplicates into validation and makes early stopping overconfident, so
    chat datasets are grouped by their raw non-system messages before splitting.
    """
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    groups = _example_groups(dataset)
    if groups is not None:
        rng = np.random.default_rng(seed)
        group_ids = rng.permutation(len(groups)).tolist()
        val_indices = []
        train_indices = []
        for group_id in group_ids:
            target = val_indices if len(val_indices) < n_val else train_indices
            target.extend(groups[group_id])
        if train_indices and val_indices:
            return train_indices, val_indices

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return indices[n_val:].tolist(), indices[:n_val].tolist()


def _raw_example_at(dataset, idx: int):
    if hasattr(dataset, "examples"):
        return dataset.examples[idx]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return _raw_example_at(dataset.dataset, int(dataset.indices[idx]))
    return None


def _example_signature(example: dict) -> str:
    messages = example.get("messages", [])
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            continue
        if role == "assistant" and msg.get("train", True) is not False:
            content = "<supervised-target>"
        else:
            content = " ".join(str(msg.get("content", "")).lower().split())
        parts.append((role, content))
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def _example_groups(dataset) -> list[list[int]] | None:
    grouped: dict[str, list[int]] = {}
    for idx in range(len(dataset)):
        example = _raw_example_at(dataset, idx)
        if not isinstance(example, dict):
            return None
        grouped.setdefault(_example_signature(example), []).append(idx)
    return list(grouped.values())


class SubsetView:
    """Lightweight subset — mirrors torch.utils.data.Subset for dataset-agnostic code."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        self._indices_array = np.asarray(self.indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def get_batch(self, indices: list[int] | np.ndarray):
        mapped = self._indices_array[np.asarray(indices, dtype=np.int64)]
        get_batch = getattr(self.dataset, "get_batch", None)
        if get_batch is not None:
            return get_batch(mapped)
        return _stack_items(self.dataset, mapped)


class ResumableBatchSamplerMLX:
    """Deterministic shuffled sampler backed by numpy's Generator for exact resume."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
        rng_state: dict | None = None,
        indices: list[int] | np.ndarray | None = None,
        position: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        self.indices = (
            np.asarray(indices, dtype=np.int64) if indices is not None else np.empty(0, dtype=np.int64)
        )
        self.position = position
        if self.indices.size == 0:
            self._refresh_indices()

    def _refresh_indices(self) -> None:
        self.indices = self.rng.permutation(self.dataset_size).astype(np.int64, copy=False)
        self.position = 0

    def __iter__(self):
        while True:
            remaining = len(self.indices) - self.position
            if remaining < self.batch_size:
                if not self.drop_last and remaining > 0:
                    batch = self.indices[self.position :]
                    self._refresh_indices()
                    yield batch
                else:
                    self._refresh_indices()
                continue
            next_position = self.position + self.batch_size
            batch = self.indices[self.position : next_position]
            self.position = next_position
            yield batch

    def advance_batches(self, count: int) -> None:
        """Advance sampler state without allocating batches that are discarded."""
        if count < 0:
            raise ValueError("count must be non-negative")
        while count:
            remaining = len(self.indices) - self.position
            if remaining < self.batch_size:
                if not self.drop_last and remaining > 0:
                    self._refresh_indices()
                    count -= 1
                else:
                    self._refresh_indices()
                continue
            self.position += self.batch_size
            count -= 1

    def iter_fixed(self, max_batches: int | None = None):
        """Iterate a stable prefix of the current permutation without mutating state.

        Validation uses this path so every evaluation measures the same examples
        and does not advance or reshuffle a persistent sampler behind the scenes.
        """
        yielded = 0
        for start in range(0, len(self.indices), self.batch_size):
            if max_batches is not None and yielded >= max_batches:
                break
            batch = self.indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            if len(batch) == 0:
                break
            yield batch
            yielded += 1

    def state_dict(self, *, copy_indices: bool = True) -> dict:
        return {
            "rng_state": self.rng.bit_generator.state,
            "indices": self.indices.copy() if copy_indices else self.indices,
            "position": self.position,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "ResumableBatchSamplerMLX":
        return cls(
            dataset_size=int(state["dataset_size"]),
            batch_size=int(state["batch_size"]),
            drop_last=bool(state.get("drop_last", True)),
            rng_state=state.get("rng_state"),
            indices=state.get("indices"),
            position=int(state.get("position", 0)),
        )


def sequence_length_at(dataset, idx: int, *, pad_id: int = 0, ignore_index: int = -100) -> int:
    x, y = dataset[idx]
    keep = (x != pad_id) | (y != ignore_index)
    if not keep.any():
        return 1
    return int(np.nonzero(keep)[0].max()) + 1


def sequence_lengths(dataset, *, pad_id: int = 0, ignore_index: int = -100) -> np.ndarray:
    direct = getattr(dataset, "sequence_lengths", None)
    if callable(direct):
        return np.asarray(direct(), dtype=np.int32)
    if isinstance(dataset, SubsetView):
        parent_lengths = sequence_lengths(
            dataset.dataset, pad_id=pad_id, ignore_index=ignore_index
        )
        return parent_lengths[dataset._indices_array]
    lengths = np.empty((len(dataset),), dtype=np.int32)
    batch_size = 4096
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        x, y = stack_batch(dataset, np.arange(start, stop, dtype=np.int64))[:2]
        keep = (x != pad_id) | (y != ignore_index)
        if keep.shape[1] == 0:
            lengths[start:stop] = 1
            continue
        has_tokens = keep.any(axis=1)
        last_from_right = np.argmax(keep[:, ::-1], axis=1)
        lengths[start:stop] = np.where(
            has_tokens,
            keep.shape[1] - last_from_right,
            1,
        ).astype(np.int32, copy=False)
    return lengths


class LengthBucketBatchSamplerMLX:
    """Infinite sortish sampler with one finite, lossless logical epoch.

    Buckets are batch-aligned so ``drop_last`` discards at most the single
    dataset-wide remainder. Earlier code discarded a remainder in every
    bucket, then the training loop consumed ``len(dataset) // batch_size``
    batches anyway and spilled into the next permutation.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        *,
        bucket_size: int = 2048,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = batch_size
        requested_bucket_size = max(batch_size, bucket_size)
        self.bucket_size = max(
            batch_size,
            (requested_bucket_size // batch_size) * batch_size,
        )
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        while True:
            order = self.rng.permutation(n)
            buckets = [
                order[start : start + self.bucket_size]
                for start in range(0, n, self.bucket_size)
            ]
            self.rng.shuffle(buckets)
            for bucket in buckets:
                sorted_bucket = bucket[np.argsort(self.lengths[bucket])]
                if self.rng.random() < 0.5:
                    sorted_bucket = sorted_bucket[::-1]
                for start in range(0, len(sorted_bucket), self.batch_size):
                    batch = sorted_bucket[start : start + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    yield batch


class SortedLengthBatchSamplerMLX:
    """Infinite globally length-sorted batches with shuffled batch order."""

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = np.argsort(self.lengths, kind="stable").astype(np.int64, copy=False)
        while True:
            batches: list[np.ndarray] = []
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
            batch_order = self.rng.permutation(len(batches))
            for batch_idx in batch_order:
                yield batches[int(batch_idx)]


class HomogeneousStepSortedBatchSamplerMLX:
    """Globally length-sorted batches grouped into same-shape optimizer steps.

    This preserves the low padding of globally sorted microbatches, but emits
    groups of ``grad_accum`` microbatches with the same rounded sequence length
    when possible. The training loop consumes those groups as one optimizer
    update, reducing compiled step-shape churn without changing model math.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        grad_accum: int,
        *,
        bucket_multiple: int = 16,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.grad_accum = max(1, int(grad_accum))
        self.bucket_multiple = max(1, int(bucket_multiple))
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def _rounded_len(self, batch: np.ndarray) -> int:
        max_len = int(self.lengths[batch].max(initial=1))
        return ((max_len + self.bucket_multiple - 1) // self.bucket_multiple) * self.bucket_multiple

    def __iter__(self):
        order = np.argsort(self.lengths, kind="stable").astype(np.int64, copy=False)
        while True:
            groups: dict[int, list[np.ndarray]] = {}
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                groups.setdefault(self._rounded_len(batch), []).append(batch)

            steps: list[list[np.ndarray]] = []
            leftovers: list[tuple[int, np.ndarray]] = []
            for rounded_len, batches in groups.items():
                full_count = (len(batches) // self.grad_accum) * self.grad_accum
                for start in range(0, full_count, self.grad_accum):
                    steps.append(batches[start : start + self.grad_accum])
                leftovers.extend((rounded_len, batch) for batch in batches[full_count:])

            leftovers.sort(key=lambda item: item[0])
            for start in range(0, len(leftovers), self.grad_accum):
                chunk = leftovers[start : start + self.grad_accum]
                if len(chunk) < self.grad_accum and self.drop_last:
                    continue
                steps.append([batch for _, batch in chunk])

            for step_idx in self.rng.permutation(len(steps)):
                for batch in steps[int(step_idx)]:
                    yield batch


class StepSortedBatchSamplerMLX:
    """Shuffle normally, then sort only within each optimizer-step window.

    This keeps the set of examples seen by each optimizer step identical to a
    plain shuffled sampler with the same seed, batch size, and grad accumulation
    count, but reduces padding within the microbatches that make up the step.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        grad_accum: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.grad_accum = max(1, int(grad_accum))
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.lengths)
        window_size = self.batch_size * self.grad_accum
        while True:
            order = self.rng.permutation(n).astype(np.int64, copy=False)
            for start in range(0, n, window_size):
                window = order[start : start + window_size]
                if len(window) < window_size and self.drop_last:
                    continue
                sorted_window = window[np.argsort(self.lengths[window])]
                for micro_start in range(0, len(sorted_window), self.batch_size):
                    batch = sorted_window[micro_start : micro_start + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    yield batch


class WindowSortedBatchSamplerMLX:
    """Shuffle normally, then sort within a multi-step window.

    This trades off convergence risk and padding efficiency between
    ``StepSortedBatchSamplerMLX`` and globally sorted batches. The set of
    examples in each window matches the plain shuffled sampler, but the order
    inside that window is length-sorted before microbatches are emitted.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        grad_accum: int,
        window_steps: int,
        *,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.grad_accum = max(1, int(grad_accum))
        self.window_steps = max(1, int(window_steps))
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = len(self.lengths)
        window_size = self.batch_size * self.grad_accum * self.window_steps
        while True:
            order = self.rng.permutation(n).astype(np.int64, copy=False)
            for start in range(0, n, window_size):
                window = order[start : start + window_size]
                if len(window) < window_size and self.drop_last:
                    continue
                sorted_window = window[np.argsort(self.lengths[window])]
                batches = []
                for micro_start in range(0, len(sorted_window), self.batch_size):
                    batch = sorted_window[micro_start : micro_start + self.batch_size]
                    if len(batch) < self.batch_size and self.drop_last:
                        continue
                    batches.append(batch)
                for batch_idx in self.rng.permutation(len(batches)):
                    yield batches[int(batch_idx)]


class TokenBudgetLengthBatchSamplerMLX:
    """Infinite length-sorted SFT sampler with a small static shape lattice.

    Each microbatch is drawn from examples that fit within one rounded sequence
    length and uses as many examples as fit under ``token_budget``. This keeps
    B*T roughly stable while limiting MLX compile shapes to the provided
    lattice.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        *,
        token_budget: int,
        lattice: tuple[int, ...] = (128, 192, 256, 320, 384, 448, 512),
        max_batch_size: int = 128,
        min_batch_size: int = 1,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.token_budget = int(token_budget)
        self.lattice = tuple(sorted(int(x) for x in lattice))
        self.max_batch_size = int(max_batch_size)
        self.min_batch_size = int(min_batch_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not self.lattice:
            raise ValueError("lattice must contain at least one sequence length")

    def __iter__(self):
        bins: dict[int, list[int]] = {length: [] for length in self.lattice}
        for idx, length in enumerate(self.lengths):
            target = self.lattice[-1]
            for candidate in self.lattice:
                if int(length) <= candidate:
                    target = candidate
                    break
            bins[target].append(idx)

        while True:
            batches: list[np.ndarray] = []
            for target_len, indices in bins.items():
                if not indices:
                    continue
                order = np.asarray(indices, dtype=np.int64)
                order = order[self.rng.permutation(len(order))]
                batch_size = max(self.min_batch_size, self.token_budget // target_len)
                batch_size = min(self.max_batch_size, max(1, batch_size))
                for start in range(0, len(order), batch_size):
                    batch = order[start : start + batch_size]
                    if len(batch) < batch_size and self.drop_last:
                        continue
                    batches.append(batch)
            batch_order = self.rng.permutation(len(batches))
            for batch_idx in batch_order:
                yield batches[int(batch_idx)]


def _stack_items(dataset, indices: list[int] | np.ndarray):
    columns = None
    for idx in indices:
        item = dataset[int(idx)]
        if not isinstance(item, tuple):
            item = (item,)
        if columns is None:
            columns = [[] for _ in item]
        for column, value in zip(columns, item):
            column.append(value)
    if columns is None:
        raise ValueError("cannot stack an empty batch")
    return tuple(np.stack(column) for column in columns)


def stack_batch(dataset, indices: list[int] | np.ndarray):
    get_batch = getattr(dataset, "get_batch", None)
    if get_batch is not None:
        return get_batch(indices)
    return _stack_items(dataset, indices)


def pack_sft_batch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pad_id: int = 0,
    ignore_index: int = -100,
    max_seq_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a logical SFT batch into fewer physical rows.

    Returns `(x, y, segment_ids, position_ids)`. Segment IDs define independent
    examples for block-causal attention; position IDs reset to zero for each
    packed example so logits match the unpacked path.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("pack_sft_batch expects matching rank-2 x/y arrays")
    logical_batch, original_seq_len = x.shape
    max_len = int(max_seq_len or original_seq_len)
    if max_len <= 0:
        raise ValueError("max_seq_len must be positive")

    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    rows_seg: list[np.ndarray] = []
    rows_pos: list[np.ndarray] = []
    row_cursors: list[int] = []
    row_segments: list[int] = []

    keep = (x != pad_id) | (y != ignore_index)
    if original_seq_len:
        has_tokens = keep.any(axis=1)
        last_from_right = np.argmax(keep[:, ::-1], axis=1)
        lengths = np.where(
            has_tokens,
            original_seq_len - last_from_right,
            0,
        ).astype(np.int32, copy=False)
        np.minimum(lengths, max_len, out=lengths)
    else:
        lengths = np.zeros((logical_batch,), dtype=np.int32)
    row_order = np.argsort(lengths)[::-1]
    positions = np.arange(max_len, dtype=np.int32)

    for row_idx in row_order:
        length = int(lengths[row_idx])
        if length <= 0:
            continue
        target = -1
        for packed_idx, cursor in enumerate(row_cursors):
            if cursor + length <= max_len:
                target = packed_idx
                break
        if target < 0:
            target = len(rows_x)
            rows_x.append(np.full((max_len,), pad_id, dtype=np.int32))
            rows_y.append(np.full((max_len,), ignore_index, dtype=np.int32))
            rows_seg.append(np.full((max_len,), -1, dtype=np.int32))
            rows_pos.append(np.zeros((max_len,), dtype=np.int32))
            row_cursors.append(0)
            row_segments.append(0)
        cursor = row_cursors[target]
        end = cursor + length
        rows_x[target][cursor:end] = x[row_idx, :length]
        rows_y[target][cursor:end] = y[row_idx, :length]
        rows_seg[target][cursor:end] = row_segments[target]
        rows_pos[target][cursor:end] = positions[:length]
        row_cursors[target] = end
        row_segments[target] += 1

    if not rows_x:
        rows_x.append(np.full((max_len,), pad_id, dtype=np.int32))
        rows_y.append(np.full((max_len,), ignore_index, dtype=np.int32))
        rows_seg.append(np.full((max_len,), -1, dtype=np.int32))
        rows_pos.append(np.zeros((max_len,), dtype=np.int32))

    return (
        np.ascontiguousarray(np.stack(rows_x)),
        np.ascontiguousarray(np.stack(rows_y)),
        np.ascontiguousarray(np.stack(rows_seg)),
        np.ascontiguousarray(np.stack(rows_pos)),
    )


def append_supervised_loss_indices(
    batch: tuple[np.ndarray, ...],
    *,
    ignore_index: int = -100,
    bucket_multiple: int = 128,
) -> tuple[np.ndarray, ...]:
    """Append fixed-shape supervised-token indices for masked SFT loss.

    MLX cannot gather rows with a dynamic boolean mask inside a compiled graph.
    The SFT mask is already known on the CPU when the batch is built, so pass
    padded integer positions plus a mask. The model can then compute the
    vocabulary projection only for supervised tokens while preserving the exact
    masked-loss value.
    """
    if len(batch) < 2:
        raise ValueError("expected at least x and y arrays")
    y = batch[1]
    if y.ndim != 2:
        return batch

    flat_y = y.reshape(-1)
    valid_positions = np.flatnonzero(flat_y != ignore_index).astype(np.int32, copy=False)
    valid_count = int(valid_positions.shape[0])
    if bucket_multiple > 1:
        padded_count = ((max(valid_count, 1) + bucket_multiple - 1) // bucket_multiple) * bucket_multiple
    else:
        padded_count = max(valid_count, 1)

    loss_indices = np.zeros((padded_count,), dtype=np.int32)
    loss_targets = np.zeros((padded_count,), dtype=np.int32)
    loss_mask = np.zeros((padded_count,), dtype=np.float32)
    if valid_count > 0:
        loss_indices[:valid_count] = valid_positions
        loss_targets[:valid_count] = flat_y[valid_positions]
        loss_mask[:valid_count] = 1.0

    return (
        *batch,
        np.ascontiguousarray(loss_indices),
        np.ascontiguousarray(loss_targets),
        np.ascontiguousarray(loss_mask),
    )


def append_valid_token_indices(
    batch: tuple[np.ndarray, ...],
    *,
    pad_id: int = 0,
    ignore_index: int = -100,
    bucket_multiple: int = 128,
) -> tuple[np.ndarray, ...]:
    """Append fixed-shape real-token row indices for compact MLP execution.

    Valid rows are every non-padding input position, including user/prompt
    tokens with ignored labels. SFT loss only supervises assistant targets, but
    prompt tokens are still causal context and must keep flowing through the
    transformer. Capacity is rounded on the CPU so compiled MLX functions avoid
    dynamic-shape masks.
    """
    if len(batch) < 2:
        raise ValueError("expected at least x and y arrays")
    x, y = batch[:2]
    if x.ndim != 2 or y.ndim != 2:
        return batch

    flat_keep = ((x != pad_id) | (y != ignore_index)).reshape(-1)
    valid_positions = np.flatnonzero(flat_keep).astype(np.int32, copy=False)
    valid_count = int(valid_positions.shape[0])
    full_rows = int(flat_keep.shape[0])
    if full_rows <= 0:
        return batch

    if bucket_multiple > 1:
        capacity = ((max(valid_count, 1) + bucket_multiple - 1) // bucket_multiple) * bucket_multiple
    else:
        capacity = max(valid_count, 1)
    capacity = min(capacity, full_rows)

    valid_indices = np.empty((capacity,), dtype=np.int32)
    valid_mask = np.zeros((capacity,), dtype=np.float32)
    if valid_count > 0:
        n = min(valid_count, capacity)
        valid_indices[:n] = valid_positions[:n]
        valid_mask[:n] = 1.0
    else:
        n = 0

    if n < capacity:
        pad_position = int(np.argmax(~flat_keep))
        valid_indices[n:] = pad_position

    return (
        *batch,
        np.ascontiguousarray(valid_indices),
        np.ascontiguousarray(valid_mask),
    )


def append_packed_varlen_attention_metadata(
    batch: tuple[np.ndarray, ...],
    *,
    ignore_index: int = -100,
) -> tuple[np.ndarray, ...]:
    """Append metadata for MFA varlen attention on packed SFT batches."""
    if len(batch) < 4:
        raise ValueError("packed varlen attention metadata requires segment_ids")
    segment_ids = batch[2]
    if segment_ids.ndim != 2:
        raise ValueError("segment_ids must be rank-2")

    _, seq_len = segment_ids.shape
    flat_segment_ids = segment_ids.reshape(-1)
    flat_indices = np.flatnonzero(flat_segment_ids >= 0).astype(np.int32, copy=False)
    if flat_indices.size:
        selected_segments = flat_segment_ids[flat_indices]
        row_ids = flat_indices.astype(np.int64, copy=False) // seq_len
        segment_starts = np.empty((flat_indices.size,), dtype=bool)
        segment_starts[0] = True
        segment_starts[1:] = (
            (row_ids[1:] != row_ids[:-1])
            | (selected_segments[1:] != selected_segments[:-1])
        )
        starts = np.flatnonzero(segment_starts)
        lengths = np.diff(
            np.append(starts, flat_indices.size)
        ).astype(np.int32, copy=False)
    else:
        flat_indices = np.asarray([0], dtype=np.int32)
        lengths = np.asarray([1], dtype=np.int32)
    cu = np.zeros((lengths.size + 1,), dtype=np.int32)
    cu[1:] = np.cumsum(lengths, dtype=np.int64)
    return (
        *batch,
        np.ascontiguousarray(flat_indices),
        np.ascontiguousarray(cu),
    )


def trim_right_padding_bucket(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pad_id: int = 0,
    ignore_index: int = -100,
    bucket_multiple: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim SFT batches to a right-padding bucket while preserving all tokens."""
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        return x, y
    keep = (x != pad_id) | (y != ignore_index)
    active_columns = np.flatnonzero(keep.any(axis=0))
    if active_columns.size == 0:
        target_len = 1
    else:
        target_len = int(active_columns[-1]) + 1
    if bucket_multiple > 1:
        target_len = (
            (target_len + bucket_multiple - 1) // bucket_multiple
        ) * bucket_multiple
    target_len = max(1, min(target_len, x.shape[1]))
    if target_len == x.shape[1]:
        return x, y
    return np.ascontiguousarray(x[:, :target_len]), np.ascontiguousarray(y[:, :target_len])


def trim_after_last_supervised_bucket(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ignore_index: int = -100,
    bucket_multiple: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim tokens that occur after the final supervised target in a batch.

    Causal SFT loss at position `j` only depends on input positions `<= j`.
    User/system tokens after the last supervised label cannot affect the loss,
    but ordinary right-padding trim keeps them because they are non-pad inputs.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        return x, y
    supervised = y != ignore_index
    active_columns = np.flatnonzero(supervised.any(axis=0))
    if active_columns.size == 0:
        target_len = 1
    else:
        target_len = int(active_columns[-1]) + 1
    if bucket_multiple > 1:
        target_len = (
            (target_len + bucket_multiple - 1) // bucket_multiple
        ) * bucket_multiple
    target_len = max(1, min(target_len, x.shape[1]))
    if target_len == x.shape[1]:
        return x, y
    return np.ascontiguousarray(x[:, :target_len]), np.ascontiguousarray(y[:, :target_len])


def iterate_batches(dataset, sampler: ResumableBatchSamplerMLX):
    """Yield stacked numpy batches from a sampler. Callers convert to mx.array."""
    for indices in sampler:
        yield stack_batch(dataset, indices)
