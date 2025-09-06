# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Helper functions to fit models in parallel."""

from __future__ import annotations

import os
from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

R = TypeVar("R", covariant=True)
C = TypeVar("C")


@dataclass
class Resampler:
    n_resamples: int = 9999
    rng: np.random.Generator = np.random.default_rng()
    batch: int | None = None
    n_jobs: int = 1

    def __post_init__(self):
        assert isinstance(self.rng, np.random.Generator)
        if self.n_jobs < 0:
            self.n_jobs = os.cpu_count() or 1
        assert self.n_resamples > 0 and isinstance(self.n_jobs, int)
        assert self.n_jobs > 0 and isinstance(self.n_jobs, int)
        if self.batch is None:
            self.batch = self.n_resamples
        assert self.n_resamples >= self.batch and isinstance(self.batch, int)

    def divide_work(self):
        batch_size, remainder = divmod(self.n_resamples, self.n_jobs)
        assert self.batch is not None, f"{self.batch=} must be an integer."
        if self.batch >= batch_size:
            yield from self._gen_work(self.n_jobs, batch_size, remainder)
        else:
            n_splits, remainder = divmod(self.n_resamples, self.batch)
            yield from self._gen_work(n_splits, self.batch, remainder)

    @classmethod
    def _gen_work(cls, n_iterations: int, batch_size: int, remainder: int):
        for _ in range(n_iterations):
            yield batch_size
        if remainder > 0:
            yield remainder


@runtime_checkable
class SupportsResampling(Protocol):
    def __call__(
        self,
        rng: np.random.Generator = np.random.default_rng(),
        n_resamples: int = 9999,
    ) -> NDArray[np.float64]: ...


def _simulate_in_parallel(
    simulate_fn: SupportsResampling,
    resampler: Resampler = Resampler(),
) -> NDArray[np.float64]:
    assert isinstance(simulate_fn, SupportsResampling)
    futures: set[Future] = set()
    results: Iterable[NDArray[np.float64]] = list()
    batch_sizes = list(resampler.divide_work())
    rngs = [rng for rng in resampler.rng.spawn(len(batch_sizes))]
    with ThreadPoolExecutor(max_workers=resampler.n_jobs) as executor:
        for rng, batch_size in zip(rngs, batch_sizes):
            if len(futures) >= resampler.n_jobs:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for f in completed:
                    results.append(f.result())
            args_run = (rng, batch_size)
            futures.add(executor.submit(simulate_fn, *args_run))
        for f in as_completed(futures):
            results.append(f.result())
    return np.concat(results, axis=None)
