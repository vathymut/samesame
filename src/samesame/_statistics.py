"""Internal statistics for source-versus-target tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve


def harmful_shift_statistic(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Compute the weighted harmful-shift statistic."""
    fpr, tpr, thresholds = roc_curve(
        actual,
        predicted,
        pos_label=None,
        sample_weight=sample_weight,
    )
    negative_mask = actual == 0
    negative_weights = None if sample_weight is None else sample_weight[negative_mask]
    cdf_values = _weighted_ecdf(
        predicted[negative_mask], thresholds, freq_weights=negative_weights
    )
    weights = np.power(cdf_values, 2)
    return float(trapezoid(y=tpr * weights, x=fpr))


def _weighted_ecdf(
    x: NDArray,
    query: NDArray,
    freq_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate the (weighted) empirical CDF of x at each point in query.

    Raises
    ------
    ValueError
        If ``x`` or ``freq_weights`` is not one-dimensional, weights have the
        wrong length, are not finite, are negative, or sum to zero.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if freq_weights is not None:
        freq_weights = np.asarray(freq_weights)
        if freq_weights.ndim != 1:
            raise ValueError("freq_weights must be one-dimensional.")
        if not np.all(np.isfinite(freq_weights)):
            raise ValueError(
                "freq_weights must contain only finite values (no NaN or inf)."
            )
        if len(freq_weights) != len(x):
            raise ValueError("freq_weights must have the same length as x.")
        if np.any(freq_weights < 0):
            raise ValueError("freq_weights must be non-negative.")
        if freq_weights.sum() == 0:
            raise ValueError("freq_weights must not be all zero.")
        order = np.argsort(x)
        x_sorted = x[order]
        w_sorted = freq_weights[order]
        x_unique, first = np.unique(x_sorted, return_index=True)
        w_sum = np.add.reduceat(w_sorted, first)
        y = np.cumsum(w_sum) / np.sum(w_sum)
        x_nodes = x_unique
    else:
        x_nodes = np.sort(x)
        y = np.linspace(1.0 / len(x_nodes), 1.0, len(x_nodes))
    xs = np.r_[-np.inf, x_nodes]
    ys = np.r_[0.0, y]
    return ys[np.searchsorted(xs, query, "right") - 1]
