# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Helper functions for the Bayesian bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _rudirichlet(
    n_rows: int,
    n_resamples: int = 9999,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """from R function:
    https://github.com/vathymut/dsos/blob/53c8e7bb5e1aef9bd093f9f7b6bf2fbb36494de5/R/bayes-factor.R#L3
    Also see function in bayesboot package
    """
    if rng is None:
        rng = np.random.default_rng()
    weights = rng.standard_exponential(size=(n_rows, n_resamples))
    weights /= weights.sum(axis=0)
    return weights * n_rows


def _bayesian_bootstrap(
    fn: Callable[[NDArray], float],
    n_rows: int,
    n_resamples: int = 9999,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    if rng is None:
        rng = np.random.default_rng()
    weights = _rudirichlet(n_rows=n_rows, n_resamples=n_resamples, rng=rng)
    return np.apply_along_axis(fn, 0, weights)


def bayesian_posterior(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable,
    n_resamples: int = 9999,
    rng: np.random.Generator | None = None,
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
    if rng is None:
        rng = np.random.default_rng()
    bw = None if base_weight is None else np.asarray(base_weight, dtype=float)

    def fn(sample_weight):
        if bw is not None:
            sample_weight = bw * sample_weight
            sample_weight = sample_weight / sample_weight.sum() * n_rows
        return metric(actual, predicted, sample_weight=sample_weight)

    return _bayesian_bootstrap(
        fn=fn,
        n_rows=n_rows,
        n_resamples=n_resamples,
        rng=rng,
    )


def _empirical_pvalue(
    observed: float,
    null_distribution: NDArray,
    alternative: Literal["less", "greater", "two-sided"],
    *,
    adjustment: Literal[0, 1] = 1,
) -> float:
    dtype = np.array(observed).dtype
    eps = 0.0 if not np.issubdtype(dtype, np.inexact) else np.finfo(dtype).eps * 100
    gamma = np.abs(eps * observed)
    n = null_distribution.shape[0]

    if alternative == "less":
        count = (null_distribution <= observed + gamma).sum(axis=0)
    elif alternative == "greater":
        count = (null_distribution >= observed - gamma).sum(axis=0)
    else:  # two-sided
        count_less = (null_distribution <= observed + gamma).sum(axis=0)
        count_greater = (null_distribution >= observed - gamma).sum(axis=0)
        count = np.minimum(count_less + adjustment, count_greater + adjustment)
        return float(np.clip(count * 2 / (n + adjustment), 0, 1))

    return float(np.clip((count + adjustment) / (n + adjustment), 0, 1))


def _bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 1,
) -> float:
    denom_bf = _empirical_pvalue(threshold, posterior, "less", adjustment=adjustment)
    num_bf = 1.0 - denom_bf
    return float(np.float64(num_bf) / np.float64(denom_bf))
