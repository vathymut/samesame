from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target

Direction = Literal["higher-is-worse", "higher-is-better"]
RandomState = int | np.random.RandomState | np.random.Generator | None


def as_numeric_vector(values: ArrayLike, *, name: str) -> NDArray:
    """Return a validated 1D numeric array."""
    vector = column_or_1d(values)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (
        np.issubdtype(vector.dtype, np.number) or np.issubdtype(vector.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    return np.asarray(vector)


def validate_binary_actual_with_predicted(
    actual: NDArray, predicted: NDArray
) -> tuple[NDArray, NDArray]:
    actual = column_or_1d(actual)
    predicted = as_numeric_vector(predicted, name="predicted")
    check_consistent_length(actual, predicted)
    if type_of_target(actual, "actual") != "binary":
        raise ValueError("Expected 'actual' to be a binary target (e.g. 0/1 labels).")
    return np.asarray(actual), predicted


def validate_direction(direction: str) -> Direction:
    if direction not in ("higher-is-worse", "higher-is-better"):
        raise ValueError(
            "direction must be one of 'higher-is-worse' or 'higher-is-better'."
        )
    return direction


def validate_and_normalise_weights(
    sample_weight: NDArray | None,
    n: int,
) -> NDArray | None:
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


def resolve_random_state(
    random_state: RandomState,
) -> np.random.Generator | np.random.RandomState:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator | np.random.RandomState):
        return random_state
    if isinstance(random_state, Integral):
        return np.random.default_rng(int(random_state))
    raise TypeError(
        "random_state must be an int, numpy.random.Generator, "
        "numpy.random.RandomState, or None."
    )


__all__ = [
    "Direction",
    "RandomState",
    "as_numeric_vector",
    "resolve_random_state",
    "validate_and_normalise_weights",
    "validate_binary_actual_with_predicted",
    "validate_direction",
]
