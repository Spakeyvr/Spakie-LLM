"""Background-thread prefetcher for NumPy batches.

Batch assembly and optional NumPy-only transforms run on a worker thread so
they overlap the previous step's GPU work. MLX arrays stay on the main thread,
which sidesteps cross-thread eager-evaluation ambiguity.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterator

import numpy as np

from training.dataset_mlx import ResumableBatchSamplerMLX, stack_batch


_SENTINEL = object()


class BatchPrefetcher:
    """Pulls batches from `sampler` on a worker thread into a bounded queue.

    The main loop calls `__next__` (or iterates) to get numpy `(x, y)` tuples.
    The producer sampler can run ahead of consumption, so training loops must
    checkpoint a separate committed cursor that advances only after optimizer
    steps. The queue bound limits memory use; it is not a resume guarantee.
    """

    def __init__(
        self,
        dataset,
        sampler: ResumableBatchSamplerMLX,
        *,
        maxsize: int = 2,
        prepare_batch: Callable[[tuple[np.ndarray, ...]], tuple[np.ndarray, ...]] | None = None,
    ):
        self._dataset = dataset
        self._sampler = sampler
        self._prepare_batch = prepare_batch
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._sampler_iter: Iterator = iter(sampler)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    batch_indices = next(self._sampler_iter)
                except StopIteration:
                    self._sampler_iter = iter(self._sampler)
                    batch_indices = next(self._sampler_iter)
                batch = stack_batch(self._dataset, batch_indices)
                if self._prepare_batch is not None:
                    batch = self._prepare_batch(batch)
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self._error = exc
            self._queue.put(_SENTINEL)

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self) -> tuple[np.ndarray, ...]:
        item = self._queue.get()
        if item is _SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopIteration
        return item

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5.0)

    def __enter__(self) -> "BatchPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
