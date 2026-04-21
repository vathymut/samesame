# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Importance-weight builders for covariate shift adaptation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from samesame._utils import validate_binary_actual_with_predicted

ContextWeightingMode = Literal[
    "source-reweighting",
    "target-reweighting",
    "double-weighting-covariate-shift-adaptation",
]


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

    Raises
    ------
    ValueError
        If ``predicted`` contains values outside ``(0, 1)``.
    ValueError
        If ``prior_ratio`` is not finite and strictly positive.
    """
    probs = np.asarray(predicted, dtype=np.float64)
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError(
            "predicted must be membership probabilities in the open interval "
            "(0, 1); got values outside this range."
        )
    if not np.isfinite(prior_ratio) or prior_ratio <= 0.0:
        raise ValueError("prior_ratio must be finite and > 0.")
    return (probs / (1.0 - probs)) * prior_ratio


def _resolve_prior_ratio(
    actual: NDArray[np.int_],
    prior_ratio: float | None,
) -> float:
    """Resolve and validate prior ratio n_tr / n_te for binary labels."""
    if prior_ratio is not None:
        if not np.isfinite(prior_ratio) or prior_ratio <= 0.0:
            raise ValueError("prior_ratio must be finite and > 0.")
        return prior_ratio

    n_tr = int(np.sum(actual == 0))
    n_te = int(np.sum(actual == 1))
    if n_tr == 0 or n_te == 0:
        raise ValueError("actual must contain both 0 and 1 labels.")
    return n_tr / n_te


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
    ValueError
        If ``actual``/``predicted`` lengths mismatch, ``actual`` is not
        binary, inferred groups are missing, or ``prior_ratio`` is invalid.

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
    >>> from samesame.importance_weights import aiw
    >>> actual = np.array([0, 0, 1, 1])
    >>> predicted = np.array([0.25, 0.4, 0.6, 0.75])
    >>> np.round(aiw(actual, predicted), 4)
    array([0.3333, 0.6667, 1.5   , 3.    ])
    >>> np.round(aiw(actual, predicted, lam=0.0), 4)
    array([1., 1., 1., 1.])
    """
    actual, predicted = validate_binary_actual_with_predicted(actual, predicted)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0, 1].")
    ratio = _resolve_prior_ratio(actual, prior_ratio)
    density_ratio = _density_ratio(predicted, prior_ratio=ratio)
    return np.power(density_ratio, lam)


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
    ValueError
        If ``actual``/``predicted`` lengths mismatch, ``actual`` is not
        binary, inferred groups are missing, or ``prior_ratio`` is invalid.

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
    >>> from samesame.importance_weights import riw
    >>> actual = np.array([0, 0, 1, 1])
    >>> predicted = np.array([0.25, 0.4, 0.6, 0.75])
    >>> np.round(riw(actual, predicted), 4)
    array([0.5, 0.8, 1.2, 1.5])
    >>> np.round(riw(actual, predicted, lam=1.0), 4)
    array([1., 1., 1., 1.])
    """
    actual, predicted = validate_binary_actual_with_predicted(actual, predicted)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0, 1].")
    ratio = _resolve_prior_ratio(actual, prior_ratio)
    density_ratio = _density_ratio(predicted, prior_ratio=ratio)
    return density_ratio / ((1.0 - lam) + lam * density_ratio)


def _inverse_riw_from_density_ratio(
    density_ratio: NDArray[np.float64],
    *,
    lam: float,
) -> NDArray[np.float64]:
    """Compute RIW in the inverse-importance direction from density ratios."""
    # Algebraically equivalent to RIW applied to 1 / r, avoiding explicit division.
    return 1.0 / (lam + (1.0 - lam) * density_ratio)


def _validate_context_mode(mode: str) -> ContextWeightingMode:
    """Validate context weighting mode for RIW-based builders."""
    valid_modes: tuple[ContextWeightingMode, ...] = (
        "source-reweighting",
        "target-reweighting",
        "double-weighting-covariate-shift-adaptation",
    )
    if mode not in valid_modes:
        listed = ", ".join(repr(item) for item in valid_modes)
        raise ValueError(f"mode must be one of {listed}.")
    return mode


def contextual_riw(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    mode: ContextWeightingMode,
    lam: float = 0.5,
    prior_ratio: float | None = None,
) -> NDArray[np.float64]:
    """
    Build context-aware RIW sample weights for shift testing.

    This function provides three academically named weighting strategies:

    - ``source-reweighting``
    - ``target-reweighting``
    - ``double-weighting-covariate-shift-adaptation``

    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary group labels where ``0`` is reference/source and ``1`` is
        candidate/target.
    predicted : NDArray
        Membership probabilities :math:`\hat{p} = P(\text{target} \mid x)`
        in the open interval ``(0, 1)``.
    mode : ContextWeightingMode
        Context-aware weighting mode.
    lam : float, optional
        RIW blending parameter in :math:`[0, 1]`, by default ``0.5``.
    prior_ratio : float or None, optional
        Ratio :math:`n_{\text{tr}} / n_{\text{te}}` for prior correction.
        If ``None``, inferred from ``actual``.

    Returns
    -------
    NDArray[np.float64]
        Per-sample weights of shape ``predicted.shape``.

    Raises
    ------
    ValueError
        If inputs are invalid or ``mode`` is unsupported.
    """
    actual, predicted = validate_binary_actual_with_predicted(actual, predicted)
    if lam < 0.0 or lam > 1.0:
        raise ValueError("lam must be in [0, 1].")
    validated_mode = _validate_context_mode(mode)
    ratio = _resolve_prior_ratio(actual, prior_ratio)
    density_ratio = _density_ratio(predicted, prior_ratio=ratio)
    source_weight = density_ratio / ((1.0 - lam) + lam * density_ratio)
    target_weight = _inverse_riw_from_density_ratio(density_ratio, lam=lam)

    weights = np.ones_like(density_ratio, dtype=np.float64)
    if validated_mode == "source-reweighting":
        weights[actual == 0] = source_weight[actual == 0]
        return weights
    if validated_mode == "target-reweighting":
        weights[actual == 1] = target_weight[actual == 1]
        return weights

    # Double-weighting emphasizes common support by reweighting both groups.
    weights[actual == 0] = source_weight[actual == 0]
    weights[actual == 1] = target_weight[actual == 1]
    return weights


__all__ = ["aiw", "riw", "contextual_riw", "ContextWeightingMode"]
