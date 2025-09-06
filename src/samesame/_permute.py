# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from samesame._parallel import Resampler, _simulate_in_parallel


@runtime_checkable
class SupportsDataPermutation(Protocol):
    def __call__(
        self,
        data: Sequence[NDArray],
        rng: np.random.Generator,
    ) -> Sequence[NDArray] | NDArray: ...


def _shuffle_first_column(
    data: Sequence[NDArray],
    rng: np.random.Generator,
):
    it = iter(data)
    first = next(it)
    rest = tuple(it)
    first_ = np.array(first, copy=True)
    first_ = rng.permutation(first_)
    return tuple((first_, *rest))


def _resample_null(
    data: Sequence[NDArray],
    statistic: Callable,
    permute: SupportsDataPermutation,
    n_resamples: int = 9999,
    rng: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.float64]:
    results = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        data_ = permute(data, rng)
        results[i] = statistic(*data_)
    return results


def permutation_null(
    data: Sequence[NDArray],
    statistic: Callable[[NDArray, NDArray], float],
    n_resamples: int = 9999,
    rng: np.random.Generator = np.random.default_rng(),
    n_jobs: int = 1,
    batch: int | None = None,
):
    resampler = Resampler(
        n_resamples=n_resamples,
        rng=rng,
        n_jobs=n_jobs,
        batch=batch,
    )
    def _resample_partial(rng=rng, n_resamples=n_resamples):
        return _resample_null(
            data=data,
            statistic=statistic,
            permute=_shuffle_first_column,
            n_resamples=n_resamples,
            rng=rng,
        )

    return _simulate_in_parallel(
        resampler=resampler,
        simulate_fn=_resample_partial,
    )
