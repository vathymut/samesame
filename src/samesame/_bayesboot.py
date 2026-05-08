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
from scipy.special import expit, logit


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


def bayes_factor(
    posterior: NDArray,
    threshold: float = 0.0,
    adjustment: Literal[0, 1] = 0,
) -> float:
    """Compute a directional Bayes factor from posterior samples.

    The Bayes factor compares posterior support for values above a threshold
    against support for values at or below that threshold.

    Parameters
    ----------
    posterior : NDArray
        An array of posterior samples.
    threshold : float, optional
        The threshold value to test against. Default is 0.0.
    adjustment : {0, 1}, optional
        Adjustment to apply to the Bayes factor calculation. Default is 0.

    Returns
    -------
    float
        Bayes factor in favour of the posterior mass being above ``threshold``.
    """
    return _bayes_factor(posterior, threshold, adjustment)


def as_bf(pvalue: NDArray | float) -> NDArray | float:
    """Convert a one-sided p-value to a Bayes factor.

    Parameters
    ----------
    pvalue : NDArray | float
        The p-value(s) to convert. Must be strictly in (0, 1).

    Returns
    -------
    NDArray | float
        Corresponding Bayes factor(s).

    Raises
    ------
    ValueError
        If any p-value is not strictly within (0, 1).
    """
    if np.any(np.logical_or(pvalue >= 1, pvalue <= 0)):
        raise ValueError("pvalue must be within the open interval (0, 1).")
    pvalue = np.clip(pvalue, 1e-10, 1 - 1e-10)
    return 1.0 / np.exp(logit(pvalue))


def as_pvalue(bayes_factor_val: float | NDArray) -> float | NDArray:
    """Convert a Bayes factor of a directional effect to a one-sided p-value.

    Parameters
    ----------
    bayes_factor_val : float | NDArray
        The Bayes factor(s) to convert. Must be strictly positive.

    Returns
    -------
    float | NDArray
        Corresponding p-value(s).

    Raises
    ------
    ValueError
        If any Bayes factor is not strictly positive.
    """
    if np.any(bayes_factor_val <= 0):
        raise ValueError("bayes_factor must be strictly positive.")
    bf_ = np.clip(bayes_factor_val, 1e-10, 1e10)
    return expit(-np.log(bf_))


__all__ = [
    "as_bf",
    "as_pvalue",
    "bayes_factor",
    "bayesian_posterior",
]
