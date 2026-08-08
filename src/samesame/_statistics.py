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
    """Evaluate the (weighted) empirical CDF of x at each point in query."""
    x = np.asarray(x)
    if freq_weights is not None:
        freq_weights = np.asarray(freq_weights)
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
