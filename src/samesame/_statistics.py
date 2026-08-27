"""Internal statistics for source-versus-target tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve


def harmful_shift_statistic(
    labels: NDArray[np.int_],
    scores: NDArray,
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Compute the weighted harmful-shift statistic."""
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=sample_weight)
    negatives = labels == 0
    negative_weights = None if sample_weight is None else sample_weight[negatives]
    negative_cdf = _weighted_ecdf(
        scores[negatives], thresholds, freq_weights=negative_weights
    )
    return float(trapezoid(y=tpr * negative_cdf**2, x=fpr))


def _weighted_ecdf(
    x: NDArray,
    query: NDArray,
    freq_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate the (weighted) empirical CDF of ``x`` at each point in ``query``.

    Raises
    ------
    ValueError
        If ``freq_weights`` has the wrong length, contains non-finite values,
        is negative, or sums to zero.
    """
    if freq_weights is None:
        freq_weights = np.ones(len(x))
    if len(freq_weights) != len(x):
        raise ValueError("freq_weights must have the same length as x.")
    if not np.all(np.isfinite(freq_weights)):
        raise ValueError(
            "freq_weights must contain only finite values (no NaN or inf)."
        )
    if np.any(freq_weights < 0):
        raise ValueError("freq_weights must be non-negative.")
    if freq_weights.sum() == 0:
        raise ValueError("freq_weights must not be all zero.")
    order = np.argsort(x)
    x_unique, first = np.unique(x[order], return_index=True)
    weight_sums = np.add.reduceat(freq_weights[order], first)
    cdf = np.cumsum(weight_sums) / np.sum(weight_sums)
    knots = np.r_[-np.inf, x_unique]
    levels = np.r_[0.0, cdf]
    return levels[np.searchsorted(knots, query, "right") - 1]
