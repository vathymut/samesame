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

from samesame._internals.ecdf import ECDFDiscrete

SHIFT_STATISTICS: dict[str, Callable[..., float]] = {
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
    """Compute the weighted area under the ROC curve."""
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


def get_shift_statistic(name: str) -> tuple[str, Callable[..., float]]:
    statistic = SHIFT_STATISTICS.get(name)
    if statistic is None:
        allowed = ", ".join(sorted(SHIFT_STATISTICS))
        raise ValueError(f"statistic must be one of {allowed}; got {name!r}.")
    return name, statistic


def requires_binary_scores(name: str) -> bool:
    return name in _BINARY_ONLY_STATISTICS


__all__ = ["SHIFT_STATISTICS", "get_shift_statistic", "requires_binary_scores", "wauc"]

