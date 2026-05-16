from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _rudirichlet(size: int, rng: np.random.Generator | np.random.RandomState) -> NDArray:
    """Sample from a uniform Dirichlet distribution."""
    return rng.dirichlet(alpha=np.ones(size))


def _bayesian_bootstrap(
    statistic: Callable[[NDArray], float],
    n_obs: int,
    *,
    n_resamples: int,
    rng: np.random.Generator | np.random.RandomState,
) -> NDArray[np.float64]:
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        draws[idx] = statistic(_rudirichlet(n_obs, rng))
    return draws


def bayesian_posterior(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable[..., float],
    *,
    n_resamples: int,
    rng: np.random.Generator | np.random.RandomState,
    base_weight: NDArray | None = None,
) -> NDArray[np.float64]:
    """Draw a Bayesian bootstrap posterior for a metric."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")

    def fn(sample_weight: NDArray) -> float:
        if base_weight is not None:
            sample_weight = sample_weight * base_weight
        return float(metric(actual, predicted, sample_weight=sample_weight))

    return _bayesian_bootstrap(
        fn,
        len(actual),
        n_resamples=n_resamples,
        rng=rng,
    )


def _empirical_pvalue(posterior: NDArray[np.float64], threshold: float) -> float:
    if posterior.ndim != 1:
        raise ValueError("posterior must be one-dimensional.")
    return float(np.mean(posterior > threshold))


def _bayes_factor(posterior: NDArray[np.float64], threshold: float) -> float:
    pvalue = _empirical_pvalue(posterior, threshold)
    if pvalue == 1.0:
        return np.inf
    return float(pvalue / (1.0 - pvalue))


def bayes_factor(posterior: NDArray, threshold: float) -> float:
    """Compute the Bayes factor from posterior draws and a threshold."""
    posterior = np.asarray(posterior, dtype=np.float64)
    if posterior.ndim != 1:
        raise ValueError("posterior must be a one-dimensional numeric array.")
    if posterior.size == 0:
        raise ValueError("posterior must not be empty.")
    return _bayes_factor(posterior, threshold)


def as_bf(pvalue: NDArray | float) -> NDArray | float:
    """Convert p-values to Bayes factors."""
    values = np.asarray(pvalue)
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("pvalue must be within the open interval (0, 1).")
    converted = (1.0 - values) / values
    if np.isscalar(pvalue):
        return float(converted)
    return converted


def as_pvalue(bayes_factor_val: float | NDArray) -> float | NDArray:
    """Convert Bayes factors to p-values."""
    values = np.asarray(bayes_factor_val)
    if np.any(values <= 0.0):
        raise ValueError("bayes_factor must be strictly positive.")
    converted = 1.0 / (1.0 + values)
    if np.isscalar(bayes_factor_val):
        return float(converted)
    return converted


__all__ = [
    "as_bf",
    "as_pvalue",
    "bayes_factor",
    "bayesian_posterior",
    "_bayes_factor",
]

