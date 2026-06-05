"""NumPy/MLX-native datasets and resumable batch iterator for the MLX backend."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenizer.train_tokenizer import SpakieTokenizer


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


class ChatSFTDatasetMLX:
    """JSONL chat dataset with loss masking on non-assistant turns (numpy arrays)."""

    def __init__(self, jsonl_path: str, tokenizer: SpakieTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        messages = self.examples[idx]["messages"]
        input_ids: list[int] = []
        labels: list[int] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                role_token = self.tokenizer.system_id
            elif role == "user":
                role_token = self.tokenizer.user_id
            elif role == "assistant":
                role_token = self.tokenizer.assistant_id
            else:
                continue

            content_ids = self.tokenizer.encode(content)
            turn_ids = [role_token] + content_ids + [self.tokenizer.eos_id]
            if role == "assistant":
                turn_labels = [-100] + content_ids + [self.tokenizer.eos_id]
            else:
                turn_labels = [-100] * len(turn_ids)

            input_ids.extend(turn_ids)
            labels.extend(turn_labels)

        input_ids = input_ids[: self.max_seq_len + 1]
        labels = labels[: self.max_seq_len + 1]

        x = input_ids[:-1]
        y = labels[1:]
        pad_len = self.max_seq_len - len(x)
        x = x + [self.tokenizer.pad_id] * pad_len
        y = y + [-100] * pad_len

        return np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32)


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
        xs = []
        ys = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            if not isinstance(item, tuple) or len(item) < 2:
                raise ValueError("pretokenized SFT dataset expects (x, y) items")
            x, y = item[:2]
            xs.append(np.asarray(x, dtype=np.int32))
            ys.append(np.asarray(y, dtype=np.int32))
            raw = _raw_example_at(dataset, idx)
            if raw is not None:
                raw_examples.append(raw)

        if xs:
            self.x = np.ascontiguousarray(np.stack(xs))
            self.y = np.ascontiguousarray(np.stack(ys))
        else:
            self.x = np.empty((0, 0), dtype=np.int32)
            self.y = np.empty((0, 0), dtype=np.int32)
        if len(raw_examples) == len(xs):
            self.examples = raw_examples

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]


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

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


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
    return np.asarray(
        [
            sequence_length_at(dataset, i, pad_id=pad_id, ignore_index=ignore_index)
            for i in range(len(dataset))
        ],
        dtype=np.int32,
    )


class LengthBucketBatchSamplerMLX:
    """Infinite sortish sampler that batches examples with similar lengths."""

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
        self.bucket_size = max(batch_size, bucket_size)
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

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


def stack_batch(dataset, indices: list[int] | np.ndarray):
    columns = None
    for idx in indices:
        item = dataset[idx]
        if not isinstance(item, tuple):
            item = (item,)
        if columns is None:
            columns = [[] for _ in item]
        for column, value in zip(columns, item):
            column.append(value)
    if columns is None:
        raise ValueError("cannot stack an empty batch")
    return tuple(np.stack(column) for column in columns)


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

    lengths = np.zeros((logical_batch,), dtype=np.int32)
    for row_idx in range(logical_batch):
        keep = (x[row_idx] != pad_id) | (y[row_idx] != ignore_index)
        if keep.any():
            lengths[row_idx] = min(int(np.nonzero(keep)[0].max()) + 1, max_len)
    row_order = np.argsort(lengths)[::-1]

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
        rows_pos[target][cursor:end] = np.arange(length, dtype=np.int32)
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
        invalid_positions = np.flatnonzero(~flat_keep).astype(np.int32, copy=False)
        pad_position = int(invalid_positions[0]) if invalid_positions.size else 0
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

    flat_indices: list[int] = []
    lengths: list[int] = []
    rows, seq_len = segment_ids.shape
    for row in range(rows):
        row_seg = segment_ids[row]
        valid = np.nonzero(row_seg >= 0)[0]
        if valid.size == 0:
            continue
        start = 0
        while start < valid.size:
            first_pos = int(valid[start])
            seg = int(row_seg[first_pos])
            stop = start + 1
            while stop < valid.size and int(row_seg[int(valid[stop])]) == seg:
                stop += 1
            positions = valid[start:stop].astype(np.int32, copy=False)
            flat_indices.extend((row * seq_len + positions).tolist())
            lengths.append(int(stop - start))
            start = stop

    if not flat_indices:
        flat_indices = [0]
        lengths = [1]
    cu = np.zeros((len(lengths) + 1,), dtype=np.int32)
    cu[1:] = np.cumsum(np.asarray(lengths, dtype=np.int32))
    return (
        *batch,
        np.ascontiguousarray(np.asarray(flat_indices, dtype=np.int32)),
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
    if not keep.any():
        target_len = 1
    else:
        target_len = int(np.nonzero(keep)[1].max()) + 1
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
    if not supervised.any():
        target_len = 1
    else:
        target_len = int(np.nonzero(supervised)[1].max()) + 1
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
