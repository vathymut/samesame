from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

WeightingMode = Literal["source", "target", "both"]


def density_ratio(
    membership_prob: NDArray,
    *,
    group_balance: float,
) -> NDArray[np.float64]:
    probs = np.asarray(membership_prob, dtype=np.float64)
    if np.any(probs <= 0.0) or np.any(probs >= 1.0):
        raise ValueError("domain probabilities must be in the open interval (0, 1).")
    if not np.isfinite(group_balance) or group_balance <= 0.0:
        raise ValueError("group_balance must be finite and > 0.")
    return (probs / (1.0 - probs)) * group_balance


def riw(density_ratio_values: NDArray, *, lam: float) -> NDArray[np.float64]:
    return density_ratio_values / ((1.0 - lam) + lam * density_ratio_values)


def inverse_riw(density_ratio_values: NDArray, *, lam: float) -> NDArray[np.float64]:
    return 1.0 / (lam + (1.0 - lam) * density_ratio_values)


def validate_mode(mode: str) -> WeightingMode:
    valid: tuple[WeightingMode, ...] = ("source", "target", "both")
    if mode not in valid:
        listed = ", ".join(repr(m) for m in valid)
        raise ValueError(f"mode must be one of {listed}.")
    return mode  # type: ignore[return-value]


__all__ = ["WeightingMode", "density_ratio", "inverse_riw", "riw", "validate_mode"]

