"""Internal source-target Outlier score preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d

from samesame.weights import ImportanceWeights


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
    return np.asarray(vector)
