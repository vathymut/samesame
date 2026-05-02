"""Sample weight builders for covariate shift adaptation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from samesame._utils import validate_binary_actual_with_predicted

WeightingMode = Literal["source", "target", "both"]


def _resolve_group_balance(group: NDArray[np.int_], balance: bool) -> float:
    if not balance:
        return 1.0
    n_source = int(np.sum(group == 0))
    n_target = int(np.sum(group == 1))
    if n_source == 0 or n_target == 0:
        raise ValueError("group must contain both 0 (source) and 1 (target) labels.")
    return n_source / n_target


def _density_ratio(
    membership_prob: NDArray,
    *,
    group_balance: float,
) -> NDArray[np.float64]:
    probs = np.asarray(membership_prob, dtype=np.float64)
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError(
            "membership_prob must be probabilities in the open interval (0, 1)."
        )
    if not np.isfinite(group_balance) or group_balance <= 0.0:
        raise ValueError("group_balance must be finite and > 0.")
    return (probs / (1.0 - probs)) * group_balance


def _riw(density_ratio: NDArray, *, alpha_blend: float) -> NDArray[np.float64]:
    return density_ratio / ((1.0 - alpha_blend) + alpha_blend * density_ratio)


def _inverse_riw(density_ratio: NDArray, *, alpha_blend: float) -> NDArray[np.float64]:
    return 1.0 / (alpha_blend + (1.0 - alpha_blend) * density_ratio)


def _validate_mode(mode: str) -> WeightingMode:
    valid: tuple[WeightingMode, ...] = ("source", "target", "both")
    if mode not in valid:
        listed = ", ".join(repr(m) for m in valid)
        raise ValueError(f"mode must be one of {listed}.")
    return mode  # type: ignore[return-value]


def contextual_weights(
    group: NDArray[np.int_],
    membership_prob: NDArray,
    *,
    mode: WeightingMode = "source",
    alpha_blend: float = 0.5,
    balance: bool = True,
) -> NDArray[np.float64]:
    """Build context-aware sample weights for shift testing.

    Computes per-sample weights from membership probabilities using Relative
    Importance Weighting (RIW). Useful when the source and target groups
    differ in their covariate distributions and that difference should be
    accounted for before comparison.

    Parameters
    ----------
    group : NDArray[np.int_]
        Binary group labels. ``0`` marks source samples; ``1`` marks target
        samples.
    membership_prob : NDArray
        Per-sample probability of belonging to the target group, in the open
        interval (0, 1). Typically the output of a classifier trained to
        distinguish source from target.
    mode : {'source', 'target', 'both'}, optional
        Which samples to reweight:

        - ``'source'``: reweight source samples only (default).
        - ``'target'``: reweight target samples only.
        - ``'both'``: reweight both groups simultaneously.

    alpha_blend : float, optional
        Blending parameter in [0, 1] controlling numerical stability, named
        as a nod to the academic literature. ``0.0`` gives plain
        density-ratio weights; ``1.0`` gives uniform weights. Default is
        ``0.5``.
    balance : bool, optional
        When ``True`` (default), correct for group size imbalance by
        inferring the source-to-target ratio from ``group``.
        When ``False``, assume equal group sizes.

    Returns
    -------
    NDArray[np.float64]
        Per-sample weights. Samples not targeted by ``mode`` receive
        weight 1.

    Raises
    ------
    ValueError
        If any value in ``membership_prob`` is outside (0, 1).
    ValueError
        If ``alpha_blend`` is outside [0, 1].
    ValueError
        If ``mode`` is not one of ``'source'``, ``'target'``, ``'both'``.
    ValueError
        If ``group`` does not contain both 0 and 1 labels (when
        ``balance=True``).

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import contextual_weights
    >>> group = np.array([0, 0, 1, 1])
    >>> membership_prob = np.array([0.25, 0.4, 0.6, 0.75])
    >>> np.round(contextual_weights(group, membership_prob), 4)
    array([0.5, 0.8, 1. , 1. ])
    >>> np.round(contextual_weights(group, membership_prob, mode="both"), 4)
    array([0.5, 0.8, 0.8, 0.5])
    """
    group, membership_prob = validate_binary_actual_with_predicted(
        group, membership_prob
    )
    if alpha_blend < 0.0 or alpha_blend > 1.0:
        raise ValueError("alpha_blend must be in [0, 1].")
    _validate_mode(mode)
    group_balance = _resolve_group_balance(group, balance)
    density_ratio = _density_ratio(membership_prob, group_balance=group_balance)
    source_w = _riw(density_ratio, alpha_blend=alpha_blend)
    target_w = _inverse_riw(density_ratio, alpha_blend=alpha_blend)
    out = np.ones_like(density_ratio, dtype=np.float64)
    if mode == "source":
        out[group == 0] = source_w[group == 0]
    elif mode == "target":
        out[group == 1] = target_w[group == 1]
    else:  # both
        out[group == 0] = source_w[group == 0]
        out[group == 1] = target_w[group == 1]
    return out


__all__ = ["WeightingMode", "contextual_weights"]
