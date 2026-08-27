"""Internal weighted two-sample permutation testing."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils import column_or_1d

from samesame.weights import ImportanceWeights

Rng = np.random.Generator | np.random.RandomState


def _resolve_rng(rng: Rng | None) -> Rng:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator | np.random.RandomState):
        return rng
    raise TypeError(
        "rng must be a numpy.random.Generator, numpy.random.RandomState, or None."
    )


def _permutation_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    metric: Callable[..., float],
    alternative: Literal["less", "greater", "two-sided"],
    n_resamples: int,
    rng: Rng | None,
    weights: ImportanceWeights | None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Validate scores and weights, then run a weighted permutation test.

    Returns
    -------
    tuple[float, float, NDArray[np.float64]]
        Observed statistic, p-value, and permutation null distribution.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    rng = _resolve_rng(rng)
    source_scores = _as_numeric_vector(source, name="source")
    target_scores = _as_numeric_vector(target, name="target")
    labels = np.concatenate(
        (
            np.zeros(source_scores.size, dtype=int),
            np.ones(target_scores.size, dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    sample_weight = (
        None
        if weights is None
        else _as_sample_weight(
            weights, n_source=source_scores.size, n_target=target_scores.size
        )
    )

    def statistic(labels: NDArray[np.int_], scores: NDArray) -> float:
        return float(metric(labels, scores, sample_weight=sample_weight))

    result = permutation_test(
        data=(labels, scores),
        statistic=statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        alternative=alternative,
        rng=rng,
    )
    return (
        float(result.statistic),
        float(result.pvalue),
        np.asarray(result.null_distribution, dtype=np.float64),
    )


def _as_sample_weight(
    weights: ImportanceWeights,
    *,
    n_source: int,
    n_target: int,
) -> NDArray[np.float64]:
    """Flatten importance weights into one array aligned to source-then-target scores."""
    source_weight = _check_group_weight_length(
        weights.source,
        expected_size=n_source,
        name="weights.source",
    )
    target_weight = _check_group_weight_length(
        weights.target,
        expected_size=n_target,
        name="weights.target",
    )
    return np.concatenate((source_weight, target_weight))


def _check_group_weight_length(
    sample_weight: NDArray[np.float64],
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    if sample_weight.shape[0] != expected_size:
        raise ValueError(
            f"{name} has wrong length: expected {expected_size}, "
            f"got {sample_weight.shape[0]}."
        )
    return sample_weight


def _as_numeric_vector(values: ArrayLike, *, name: str) -> NDArray:
    vector = column_or_1d(values)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (
        np.issubdtype(vector.dtype, np.number) or np.issubdtype(vector.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf).")
    return vector
