"""Harmful-shift statistic: weighted AUC with source-anchored weighting.

Implements ``∫ TPR·(1−FPR)² dFPR`` = ``∫ TPR·F_source(t)² dFPR``
(Kamulete, 2022) via ``(1−FPR) = F_source``; see
:doc:`How the harm test works <../explanation/harmful-shift-statistic>`.

References
----------
Kamulete, V. M. (2022). Test for non-negligible adverse shifts.
    *Proceedings of the 38th UAI*, PMLR 180:959-968. arXiv:2107.02990.
"""

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
    """Directional shift statistic: ``∫ TPR·(1−FPR)² dFPR`` = ``∫ TPR·F_source(t)² dFPR``.

    Larger values mean the target distribution has more mass above high
    thresholds that source rarely exceeds. Since ``1−FPR(t) = F_source(t)``,
    the ``(1−FPR)²`` form used in the docs and the ``F_source²`` form are
    identical. Inputs are the pooled labels (0 = source, 1 = target) and
    scores pooled in the same order.

    References
    ----------
    Kamulete, V. M. (2022). Test for non-negligible adverse shifts.
        *Proceedings of the 38th UAI*, PMLR 180:959-968.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=sample_weight)

    is_source = labels == 0
    source_weights = None if sample_weight is None else sample_weight[is_source]
    source_cdf = _weighted_ecdf(
        scores[is_source], thresholds, freq_weights=source_weights
    )
    return float(trapezoid(y=tpr * source_cdf**2, x=fpr))


def _weighted_ecdf(
    x: NDArray,
    query: NDArray,
    freq_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Weighted ECDF of ``x`` evaluated at ``query``.

    For each threshold ``q`` returns the total weight of observations with
    ``x ≤ q`` divided by total weight. Uses right-continuous step function.
    """
    if freq_weights is None:
        freq_weights = np.ones(len(x), dtype=np.float64)
    else:
        freq_weights = np.asarray(freq_weights, dtype=np.float64)
        if freq_weights.ndim != 1:
            raise ValueError("freq_weights must be one-dimensional.")
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

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")
    if query.ndim != 1:
        raise ValueError("query must be one-dimensional.")

    # Sort and collapse duplicates so the ECDF is constant between distinct values.
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = freq_weights[order]

    # Unique values + sum of weights per distinct value
    x_unique, first_idx = np.unique(x_sorted, return_index=True)
    weight_per_unique = np.add.reduceat(w_sorted, first_idx)
    total = float(weight_per_unique.sum())
    cdf_at_unique = np.cumsum(weight_per_unique) / total

    # Step function: 0 below min, cdf_at_unique between values, 1 above max.
    knots = np.r_[-np.inf, x_unique]
    levels = np.r_[0.0, cdf_at_unique]
    pos = np.searchsorted(knots, query, side="right") - 1
    return levels[pos]


__all__ = ["_weighted_ecdf", "harmful_shift_statistic"]
