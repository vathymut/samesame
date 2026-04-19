# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Importance weighting schemes for covariate shift adaptation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _density_ratio(
    predicted: NDArray,
    *,
    prior_ratio: float,
) -> NDArray[np.float64]:
    """Compute the prior-corrected density ratio from membership probabilities.

    Parameters
    ----------
    predicted : NDArray
        Membership probabilities in the open interval (0, 1).
    prior_ratio : float
        Ratio n_tr / n_te used to correct for class imbalance.

    Returns
    -------
    NDArray[np.float64]
        Per-sample density ratio r(x) = (p / (1 - p)) * prior_ratio.
    """
    p = np.asarray(predicted, dtype=np.float64)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError(
            "predicted must be membership probabilities in the open interval "
            "(0, 1); got values outside this range."
        )
    return (p / (1.0 - p)) * prior_ratio


def aiw(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    lam: float = 1.0,
    prior_ratio: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute Adaptive Importance Weights (AIWERM) per sample.

    AIWERM stabilises plain importance weighting (IWERM) by raising the
    density ratio to a power :math:`\\lambda \\in [0, 1]`. Setting
    :math:`\\lambda = 1` recovers exact density-ratio weighting (IWERM);
    setting :math:`\\lambda = 0` yields uniform weights.

    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary group labels. ``0`` indicates the training/reference group;
        ``1`` indicates the test/target group.
    predicted : NDArray
        Membership probabilities :math:`\\hat{p} = P(\\text{test} \\mid x)
        \\in (0, 1)`, as output by a binary classifier trained to distinguish
        the two groups. **Raw scores or logits are not accepted**; the array
        must contain calibrated probabilities strictly between 0 and 1.
    lam : float, optional
        Exponent applied to the density ratio, by default ``1.0``.
        Must be in :math:`[0, 1]`.
    prior_ratio : float or None, optional
        Ratio :math:`n_{\\text{tr}} / n_{\\text{te}}` used to correct the
        density ratio for class imbalance. When ``None`` (default), the ratio
        is inferred automatically from ``actual``. Pass ``prior_ratio=1.0``
        to assume balanced groups and disable the correction.

    Returns
    -------
    NDArray[np.float64]
        Per-sample importance weights of shape ``predicted.shape``.
        Weights are raw and unnormalized; normalization is the caller's
        responsibility.

    Raises
    ------
    ValueError
        If any element of ``predicted`` is :math:`\\le 0` or :math:`\\ge 1`.
    ValueError
        If ``lam`` is outside :math:`[0, 1]`.

    Notes
    -----
    Let :math:`r(x_i) = \\tfrac{\\hat{p}_i}{1 - \\hat{p}_i} \\cdot
    \\tfrac{n_{\\text{tr}}}{n_{\\text{te}}}` be the prior-corrected density
    ratio. The adaptive importance weight is:

    .. math::

        w_i = r(x_i)^\\lambda, \\quad \\lambda \\in [0, 1].

    The prior correction :math:`n_{\\text{tr}} / n_{\\text{te}}` accounts for
    the fact that a classifier trained on imbalanced groups learns the
    posterior :math:`P(\\text{test} \\mid x)` under the pooled prior, not the
    true density ratio. When ``prior_ratio=None``, this correction is inferred
    from ``actual``.

    References
    ----------
    .. [1] Shimodaira, H. "Improving predictive inference under covariate
       shift by weighting the log-likelihood function." Journal of Statistical
       Planning and Inference, 90(2), 2000, pp. 227-244.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.iw import aiw
    >>> actual = np.array([0, 0, 1, 1])
    >>> predicted = np.array([0.25, 0.4, 0.6, 0.75])
    >>> np.round(aiw(actual, predicted), 4)
    array([0.3333, 0.6667, 1.5   , 3.    ])
    >>> np.round(aiw(actual, predicted, lam=0.0), 4)
    array([1., 1., 1., 1.])
    """
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0, 1].")
    if prior_ratio is None:
        n_tr = int(np.sum(actual == 0))
        n_te = int(np.sum(actual == 1))
        prior_ratio = n_tr / n_te
    r = _density_ratio(predicted, prior_ratio=prior_ratio)
    return np.power(r, lam)


def riw(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    lam: float = 0.5,
    prior_ratio: float | None = None,
) -> NDArray[np.float64]:
    """
    Compute Relative Importance Weights (RIWERM) per sample.

    RIWERM replaces the density-ratio denominator with a convex combination
    of the two distributions, trading a small bias for substantially improved
    numerical stability over IWERM and AIWERM.

    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary group labels. ``0`` indicates the training/reference group;
        ``1`` indicates the test/target group.
    predicted : NDArray
        Membership probabilities :math:`\\hat{p} = P(\\text{test} \\mid x)
        \\in (0, 1)`, as output by a binary classifier trained to distinguish
        the two groups. **Raw scores or logits are not accepted**; the array
        must contain calibrated probabilities strictly between 0 and 1.
    lam : float, optional
        Blending parameter, by default ``0.5``.  Must be in :math:`[0, 1]`.
        Setting ``lam=0.0`` recovers plain density-ratio weighting (IWERM);
        setting ``lam=1.0`` yields uniform weights (all ones).
    prior_ratio : float or None, optional
        Ratio :math:`n_{\\text{tr}} / n_{\\text{te}}` used to correct the
        density ratio for class imbalance. When ``None`` (default), the ratio
        is inferred automatically from ``actual``. Pass ``prior_ratio=1.0``
        to assume balanced groups and disable the correction.

    Returns
    -------
    NDArray[np.float64]
        Per-sample importance weights of shape ``predicted.shape``.
        Weights are raw and unnormalized; normalization is the caller's
        responsibility.

    Raises
    ------
    ValueError
        If any element of ``predicted`` is :math:`\\le 0` or :math:`\\ge 1`.
    ValueError
        If ``lam`` is outside :math:`[0, 1]`.

    Notes
    -----
    Let :math:`r(x_i) = \\tfrac{\\hat{p}_i}{1 - \\hat{p}_i} \\cdot
    \\tfrac{n_{\\text{tr}}}{n_{\\text{te}}}` be the prior-corrected density
    ratio. The relative importance weight is:

    .. math::

        w_i = \\frac{r(x_i)}{(1 - \\lambda) + \\lambda\\, r(x_i)},
        \\quad \\lambda \\in [0, 1].

    At :math:`\\lambda = 0`, this reduces to :math:`w_i = r(x_i)` (plain
    IWERM). At :math:`\\lambda = 1`, the :math:`r` terms cancel and
    :math:`w_i = 1` (uniform weights). The denominator blending ensures the
    weights remain finite even when :math:`r` is large, which is the primary
    stability advantage of RIWERM.

    The prior correction :math:`n_{\\text{tr}} / n_{\\text{te}}` accounts for
    the fact that a classifier trained on imbalanced groups learns the
    posterior :math:`P(\\text{test} \\mid x)` under the pooled prior, not the
    true density ratio. When ``prior_ratio=None``, this correction is inferred
    from ``actual``.

    References
    ----------
    .. [1] Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., and Sugiyama,
       M. "Relative density-ratio estimation for robust distribution
       comparison." Neural Computation, 25(5), 2013, pp. 1324-1370.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.iw import riw
    >>> actual = np.array([0, 0, 1, 1])
    >>> predicted = np.array([0.25, 0.4, 0.6, 0.75])
    >>> np.round(riw(actual, predicted), 4)
    array([0.5, 0.8, 1.2, 1.5])
    >>> np.round(riw(actual, predicted, lam=1.0), 4)
    array([1., 1., 1., 1.])
    """
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0, 1].")
    if prior_ratio is None:
        n_tr = int(np.sum(actual == 0))
        n_te = int(np.sum(actual == 1))
        prior_ratio = n_tr / n_te
    r = _density_ratio(predicted, prior_ratio=prior_ratio)
    return r / ((1.0 - lam) + lam * r)


__all__ = ["aiw", "riw"]
