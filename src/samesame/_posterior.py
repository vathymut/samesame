"""Internal posterior evidence helpers for harmful-shift testing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from samesame._comparison import RandomNumberGenerator


def compute_posterior_evidence(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable[..., float],
    *,
    threshold: float,
    n_resamples: int,
    rng: RandomNumberGenerator,
    base_weight: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Compute Bayesian posterior and Bayes factor."""
    def statistic(sample_weight: NDArray[np.float64]) -> float:
        if base_weight is not None:
            sample_weight = sample_weight * base_weight
        return float(metric(actual, predicted, sample_weight=sample_weight))

    posterior = np.asarray(
        _bayesian_bootstrap(
            statistic,
            len(actual),
            n_resamples=n_resamples,
            rng=rng,
        ),
        dtype=np.float64,
    )
    return posterior, float(_bayes_factor(posterior, threshold))


def _draw_uniform_dirichlet(
    size: int,
    rng: RandomNumberGenerator,
) -> NDArray:
    return rng.dirichlet(alpha=np.ones(size))


def _bayesian_bootstrap(
    statistic: Callable[[NDArray], float],
    n_obs: int,
    *,
    n_resamples: int,
    rng: RandomNumberGenerator,
) -> NDArray[np.float64]:
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        draws[idx] = statistic(_draw_uniform_dirichlet(n_obs, rng))
    return draws


def _bayes_factor(posterior: NDArray[np.float64], threshold: float) -> float:
    if posterior.ndim != 1:
        raise ValueError("posterior must be one-dimensional.")
    pvalue = float(np.mean(posterior > threshold))
    if pvalue == 1.0:
        return np.inf
    return float(pvalue / (1.0 - pvalue))
