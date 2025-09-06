# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Compute the weighted area under the ROC."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve

from samesame._data import group_by
from samesame._wecdf import ECDFDiscrete


def wauc(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    """
    Compute the weighted area under the ROC curve (WAUC).

    Calculates the WAUC by weighting the true positive rate (TPR) at each
    false positive rate (FPR) threshold, optionally using sample weights.
    The weights are computed as the squared empirical weighted cumulative
    distribution function (EW-CDF) of the predicted scores for the negative
    class.

    Parameters
    ----------
    actual : NDArray[np.int_]
        Ground truth binary labels (0 or 1).
    predicted : NDArray
        Predicted scores or probabilities.
    sample_weight : NDArray or None, optional
        Sample weights. If None, all samples are given equal weight.

    Returns
    -------
    float
        The weighted area under the ROC curve.

    Notes
    -----
    The function uses the `roc_curve` from scikit-learn to compute FPR, TPR,
    and thresholds. The empirical weighted CDF is computed for the negative
    class predictions using `ECDFDiscrete`. The WAUC is calculated using the
    trapezoidal rule, weighting the TPR by the squared EW-CDF at each
    threshold [1].

    References
    ----------
    .. [1] Li, Jialiang, and Jason P. Fine. "Weighted Area under the Receiver
       Operating Characteristic Curve and Its Application to Gene Selection."
       Journal of the Royal Statistical Society: Series C (Applied Statistics),
       vol. 59, no. 4, 2010, pp. 673-692.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.metrics import wauc
    >>> actual = np.array([0, 1, 0, 1])
    >>> predicted = np.array([0.1, 0.4, 0.35, 0.8])
    >>> wauc(actual, predicted)
    np.float64(0.625)
    """
    fpr, tpr, thresholds = roc_curve(
        actual,
        predicted,
        pos_label=None,
        sample_weight=sample_weight,
    )
    if sample_weight is not None:
        stacked = np.column_stack((predicted, sample_weight))
        grp_0 = group_by(data=stacked, groups=actual)[0]
        ewcdf = ECDFDiscrete(grp_0[:, 0], freq_weights=grp_0[:, 1])
    else:
        grp_0 = group_by(data=predicted, groups=actual)[0]
        ewcdf = ECDFDiscrete(grp_0)
    weights = np.power(ewcdf(thresholds), 2)
    return trapezoid(
        y=tpr * weights,
        x=fpr,
    )
