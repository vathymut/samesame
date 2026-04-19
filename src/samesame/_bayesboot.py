# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Helper functions for the Bayesian bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _rudirichlet(
    n_rows: int,
    n_resamples: int = 9999,
    rng: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.float64]:
    """from R function:
    https://github.com/vathymut/dsos/blob/53c8e7bb5e1aef9bd093f9f7b6bf2fbb36494de5/R/bayes-factor.R#L3
    Also see function in bayesboot package
    """
    weights = rng.standard_exponential(size=(n_rows, n_resamples))
    weights /= weights.sum(axis=0)
    return weights * n_rows


def _bayesian_bootstrap(
    fn: Callable[[NDArray], float],
    n_rows: int,
    n_resamples: int = 9999,
    rng: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.float64]:
    weights = _rudirichlet(n_rows=n_rows, n_resamples=n_resamples, rng=rng)
    return np.apply_along_axis(fn, 0, weights)


def bayesian_posterior(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable,
    n_resamples: int = 9999,
    rng: np.random.Generator = np.random.default_rng(),
    base_weight: NDArray | None = None,
):
    """Compute the Bayesian bootstrap posterior of a metric.

    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary labels.
    predicted : NDArray
        Predicted scores.
    metric : Callable
        Metric conforming to scikit-learn API (accepts sample_weight kwarg).
    n_resamples : int, optional
        Number of bootstrap resamples, by default 9999.
    rng : np.random.Generator, optional
        Random number generator, by default np.random.default_rng().
    base_weight : NDArray or None, optional
        Fixed user-supplied weights (e.g. density-ratio weights). When
        provided, each Dirichlet draw is multiplied element-wise by
        base_weight and renormalised before being passed to metric.
        When None, Dirichlet draws are used directly (current behaviour).
    """
    n_rows = len(actual)

    if base_weight is None:

        def fn(sample_weight):
            return metric(actual, predicted, sample_weight=sample_weight)
    else:
        _bw = np.asarray(base_weight, dtype=float)

        def fn(sample_weight):
            w = _bw * sample_weight
            w = w / w.sum() * n_rows
            return metric(actual, predicted, sample_weight=w)

    return _bayesian_bootstrap(
        fn=fn,
        n_rows=n_rows,
        n_resamples=n_resamples,
        rng=rng,
    )
