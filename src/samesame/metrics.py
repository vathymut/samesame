# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Reference metric functions used by two-sample tests."""

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

    WAUC is an ROC-type summary that upweights operating regions according to
    the score distribution in the negative class. It is designed for settings
    where adverse high-score tails are particularly important.

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

    Raises
    ------
    ValueError
        Propagated from the underlying metric components when inputs are
        inconsistent (for example, incompatible lengths).

    Notes
    -----
    Let :math:`F_0` denote the empirical (weighted) CDF of negative-class
    scores. This implementation uses :math:`F_0(t)^2` as threshold-dependent
    weights and computes the integral by the trapezoidal rule over ROC points
    returned by :func:`sklearn.metrics.roc_curve` [1].

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
    >>> np.round(
    ...     wauc(actual, predicted, sample_weight=np.array([1.0, 2.0, 1.0, 2.0])),
    ...     3,
    ... )
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
