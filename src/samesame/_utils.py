# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target

Direction = Literal["higher-is-worse", "higher-is-better"]


def as_numeric_vector(values: ArrayLike, *, name: str) -> NDArray:
    """Return a validated 1D numeric array.

    Parameters
    ----------
    values : ArrayLike
            Input values expected to represent outlier scores.
    name : str
        Public parameter name used in validation messages.

    Returns
    -------
    NDArray
        One-dimensional numeric array.
    """
    vector = column_or_1d(values)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (
        np.issubdtype(vector.dtype, np.number) or np.issubdtype(vector.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    return np.asarray(vector)


def validate_binary_actual_with_predicted(
    actual: NDArray,
    predicted: NDArray,
) -> tuple[NDArray, NDArray]:
    """Validate binary labels against aligned outlier scores."""
    actual = column_or_1d(actual)
    predicted = as_numeric_vector(predicted, name="predicted")
    check_consistent_length(actual, predicted)
    if type_of_target(actual, "actual") != "binary":
        raise ValueError("Expected 'actual' to be a binary target (e.g. 0/1 labels).")
    return np.asarray(actual), predicted


def validate_direction(direction: str) -> Direction:
    """Validate the score direction for adverse-shift testing."""
    if direction not in ("higher-is-worse", "higher-is-better"):
        raise ValueError(
            "direction must be one of 'higher-is-worse' or 'higher-is-better'."
        )
    return direction


def validate_and_normalise_weights(
    sample_weight: NDArray | None,
    n: int,
) -> NDArray | None:
    """Validate and normalise sample weights to sum to n.

    Parameters
    ----------
    sample_weight : NDArray or None
        Sample weights to validate. If None, returns None (equal weights).
    n : int
        Expected length of the weight array.

    Returns
    -------
    NDArray or None
        Normalised weights summing to n, or None if input is None.

    Raises
    ------
    ValueError
        If weights have wrong length, contain negative values, are all zero,
        or contain non-finite values (NaN or inf).
    """
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=float)
    if len(w) != n:
        raise ValueError(f"sample_weight has wrong length: expected {n}, got {len(w)}.")
    if not np.all(np.isfinite(w)):
        raise ValueError(
            "sample_weight must contain only finite values (no NaN or inf)."
        )
    if np.any(w < 0):
        raise ValueError("sample_weight must not contain negative values.")
    total = w.sum()
    if total == 0:
        raise ValueError("sample_weight must not be all zero.")
    return w / total * n


__all__ = [
    "Direction",
    "as_numeric_vector",
    "validate_and_normalise_weights",
    "validate_binary_actual_with_predicted",
    "validate_direction",
]
