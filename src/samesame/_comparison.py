"""Internal source-target Outlier score comparison machinery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils import column_or_1d

from samesame.weights import ImportanceWeights

RandomNumberGenerator = np.random.Generator | np.random.RandomState


@dataclass(frozen=True)
class PreparedTwoSampleTest:
    """Prepared arrays shared by source-versus-target tests."""

    labels: NDArray[np.int_]
    scores: NDArray
    sample_weight: NDArray[np.float64] | None


def prepare_two_sample_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    weights: ImportanceWeights | None,
) -> PreparedTwoSampleTest:
    """Prepare validated arrays for source-versus-target testing."""
    source_scores = _as_numeric_vector(source, name="source")
    target_scores = _as_numeric_vector(target, name="target")
    labels = np.concatenate(
        (
            np.zeros(source_scores.shape[0], dtype=int),
            np.ones(target_scores.shape[0], dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    sample_weight = None
    if weights is not None:
        sample_weight = weights._as_sample_weight(
            n_source=int(source_scores.shape[0]),
            n_target=int(target_scores.shape[0]),
        )
    return PreparedTwoSampleTest(
        labels=labels,
        scores=scores,
        sample_weight=sample_weight,
    )


def run_permutation_test(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable[..., float],
    *,
    n_resamples: int,
    alternative: Literal["less", "greater", "two-sided"],
    rng: RandomNumberGenerator,
    sample_weight: NDArray[np.float64] | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Run a weighted two-sample permutation test on prepared arrays.

    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary group labels (0 for source, 1 for target).
    predicted : NDArray
        Outlier scores to test.
    metric : Callable[..., float]
        Two-sample score function accepting ``labels``, ``scores`` and
        ``sample_weight``.
    n_resamples : int
        Number of permutation resamples.
    alternative : {'less', 'greater', 'two-sided'}
        Alternative hypothesis for the permutation test.
    rng : np.random.Generator | np.random.RandomState
        Random number generator for the permutations.
    sample_weight : NDArray[np.float64] | None, optional
        Per-observation weights; ``None`` for unweighted.

    Returns
    -------
    tuple[float, float, NDArray[np.float64]]
        Observed statistic, p-value, and permutation null distribution.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not a positive integer.
    ValueError
        If ``alternative`` is not one of ``'less'``, ``'greater'``,
        ``'two-sided'``.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    def statistic(labels: NDArray[np.int_], scores: NDArray) -> float:
        return float(metric(labels, scores, sample_weight=sample_weight))

    result = permutation_test(
        data=(actual, predicted),
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
