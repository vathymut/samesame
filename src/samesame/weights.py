"""Sample weight builders for covariate shift adaptation."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

WeightingMode = Literal["source", "target", "both"]


def _density_ratio(
    membership_prob: NDArray,
    *,
    group_balance: float,
) -> NDArray[np.float64]:
    probs = np.asarray(membership_prob, dtype=np.float64)
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError(
            "membership probabilities must be in the open interval (0, 1)."
        )
    if not np.isfinite(group_balance) or group_balance <= 0.0:
        raise ValueError("group_balance must be finite and > 0.")
    return (probs / (1.0 - probs)) * group_balance


def _riw(density_ratio: NDArray, *, lam: float) -> NDArray[np.float64]:
    return density_ratio / ((1.0 - lam) + lam * density_ratio)


def _inverse_riw(density_ratio: NDArray, *, lam: float) -> NDArray[np.float64]:
    return 1.0 / (lam + (1.0 - lam) * density_ratio)


def _validate_mode(mode: str) -> WeightingMode:
    valid: tuple[WeightingMode, ...] = ("source", "target", "both")
    if mode not in valid:
        listed = ", ".join(repr(m) for m in valid)
        raise ValueError(f"mode must be one of {listed}.")
    return mode  # type: ignore[return-value]


def contextual_weights(
    *,
    source_prob: NDArray,
    target_prob: NDArray,
    mode: WeightingMode = "source",
    lambda_: float = 0.5,
) -> NDArray[np.float64]:
    """Build context-aware sample weights for shift testing.

    Computes per-sample RIW weights from context membership probabilities.
    The prior ratio is always inferred from the lengths of ``source_prob``
    and ``target_prob``.

    Parameters
    ----------
    source_prob : NDArray
        Per-sample probability of belonging to the target group for source
        samples, in the open interval (0, 1).
    target_prob : NDArray
        Per-sample probability of belonging to the target group for target
        samples, in the open interval (0, 1).
    mode : {'source', 'target', 'both'}, optional
        Which samples to reweight:

        - ``'source'``: reweight source samples only (default).
        - ``'target'``: reweight target samples only.
        - ``'both'``: reweight both groups simultaneously.

    lambda_ : float, optional
        RIW blending parameter in [0, 1]. ``0.0`` gives plain density-ratio
        weights; ``1.0`` gives uniform weights. Default is ``0.5``.

    Returns
    -------
    NDArray[np.float64]
        Per-sample weights in ``[source_weights..., target_weights...]`` order,
        matching the layout of :func:`~samesame._data.build_two_sample_dataset`.
        Samples not targeted by ``mode`` receive weight 1.

    Raises
    ------
    ValueError
        If any value in ``source_prob`` or ``target_prob`` is outside (0, 1).
    ValueError
        If ``lambda_`` is outside [0, 1].
    ValueError
        If ``mode`` is not one of ``'source'``, ``'target'``, ``'both'``.
    ValueError
        If ``source_prob`` or ``target_prob`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import contextual_weights
    >>> source_prob = np.array([0.25, 0.4])
    >>> target_prob = np.array([0.6, 0.75])
    >>> np.round(contextual_weights(source_prob=source_prob, target_prob=target_prob), 4)
    array([0.5, 0.8, 1. , 1. ])
    >>> np.round(contextual_weights(source_prob=source_prob, target_prob=target_prob, mode="both"), 4)
    array([0.5, 0.8, 0.8, 0.5])
    """
    source_prob = np.asarray(source_prob, dtype=np.float64)
    target_prob = np.asarray(target_prob, dtype=np.float64)
    n_source = len(source_prob)
    n_target = len(target_prob)
    if n_source == 0 or n_target == 0:
        raise ValueError("source_prob and target_prob must both be non-empty.")
    if lambda_ < 0.0 or lambda_ > 1.0:
        raise ValueError("lambda_ must be in [0, 1].")
    _validate_mode(mode)
    group_balance = n_source / n_target
    source_dr = _density_ratio(source_prob, group_balance=group_balance)
    target_dr = _density_ratio(target_prob, group_balance=group_balance)
    out_source = np.ones(n_source, dtype=np.float64)
    out_target = np.ones(n_target, dtype=np.float64)
    if mode in ("source", "both"):
        out_source = _riw(source_dr, lam=lambda_)
    if mode in ("target", "both"):
        out_target = _inverse_riw(target_dr, lam=lambda_)
    return np.concatenate([out_source, out_target])


__all__ = ["WeightingMode", "contextual_weights"]
