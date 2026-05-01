from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from samesame._wecdf import ECDFDiscrete

SHIFT_STATISTICS: dict[str, Callable] = {
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "matthews_corrcoef": matthews_corrcoef,
}

_BINARY_ONLY_STATISTICS = frozenset({"balanced_accuracy", "matthews_corrcoef"})


def wauc(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """Compute the weighted area under the ROC curve (WAUC)."""
    fpr, tpr, thresholds = roc_curve(
        actual,
        predicted,
        pos_label=None,
        sample_weight=sample_weight,
    )
    negative_mask = actual == 0
    negative_scores = predicted[negative_mask]
    if sample_weight is None:
        ewcdf = ECDFDiscrete(negative_scores)
    else:
        negative_weights = sample_weight[negative_mask]
        ewcdf = ECDFDiscrete(negative_scores, freq_weights=negative_weights)
    weights = np.power(ewcdf(thresholds), 2)
    return float(trapezoid(y=tpr * weights, x=fpr))


def get_shift_metric(name: str) -> tuple[str, Callable]:
    """Return a built-in shift metric by stable public name."""
    metric = SHIFT_STATISTICS.get(name)
    if metric is None:
        allowed = ", ".join(sorted(SHIFT_STATISTICS))
        raise ValueError(f"statistic must be one of {allowed}; got {name!r}.")
    return name, metric


def requires_binary_scores(name: str) -> bool:
    """Return whether the named metric requires binary outlier scores."""
    return name in _BINARY_ONLY_STATISTICS


__all__ = [
    "SHIFT_STATISTICS",
    "get_shift_metric",
    "requires_binary_scores",
    "wauc",
]
