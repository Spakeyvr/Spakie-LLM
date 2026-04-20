"""Background-thread prefetcher for numpy batches.

The training loop spends non-trivial time in `stack_batch` (numpy advanced
indexing + np.stack). Running that on a worker thread overlaps data
preparation with the previous step's GPU work. We deliberately only pass
numpy arrays across the thread boundary — MLX arrays stay on the main
thread, which sidesteps any cross-thread eager-eval ambiguity.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterator

import numpy as np

from training.dataset_mlx import ResumableBatchSamplerMLX, stack_batch


_SENTINEL = object()


class BatchPrefetcher:
    """Pulls batches from `sampler` on a worker thread into a bounded queue.

    The main loop calls `__next__` (or iterates) to get numpy `(x, y)` tuples.
    Checkpoint resume still works: the underlying sampler's state is updated
    as the worker consumes batches, and `state_dict()` on the sampler at
    checkpoint time captures a valid — possibly slightly-ahead-of-consumed —
    resume point. The maxsize=2 bound keeps that gap tiny.
    """

    def __init__(
        self,
        dataset,
        sampler: ResumableBatchSamplerMLX,
        *,
        maxsize: int = 2,
    ):
        self._dataset = dataset
        self._sampler = sampler
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

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
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
