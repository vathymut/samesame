# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import inspect
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target

Output = TypeVar("Output", bound=float | ArrayLike | NDArray | Any, covariant=True)
Input = TypeVar("Input", bound=float | ArrayLike | NDArray | Any, contravariant=True)


# Create protocols for checking
@runtime_checkable
class SupportsFitting(Protocol[Input]):
    def fit(
        self,
        X: Input,
        y=None,
        **kwargs,
    ):
        """Assumes that .fit method exists."""
        ...


@runtime_checkable
class SupportsImputer(SupportsFitting, Protocol[Input, Output]):
    def transform(
        self,
        X: Input,
    ) -> Output:
        """Assumes that .transform exists."""
        ...


@runtime_checkable
class SupportsClassifier(SupportsFitting, Protocol[Input, Output]):
    def predict_proba(
        self,
        X: Input,
    ) -> Output:
        """Assumes that .predict_proba exists."""
        ...


@runtime_checkable
class SupportsRegressor(SupportsFitting, Protocol[Input, Output]):
    def predict(
        self,
        X: Input,
    ) -> Output:
        """Assumes that .predict exists."""
        ...


@runtime_checkable
class SupportsScorer(SupportsFitting, Protocol[Input, Output]):
    def score_samples(
        self,
        X: Input,
    ) -> Output:
        """Assumes that .score_samples exists."""
        ...


SupportsEstimator = (
    SupportsClassifier | SupportsRegressor | SupportsScorer | SupportsImputer
)


def validate_binary_actual_with_predicted(
    actual: NDArray,
    predicted: NDArray,
) -> tuple[NDArray, NDArray]:
    """Validate binary actual labels against a 1D predicted-like array.

    Parameters
    ----------
    actual : NDArray
        Binary labels (e.g., 0/1).
    predicted : NDArray
        A 1D array aligned with ``actual``.

    Returns
    -------
    tuple[NDArray, NDArray]
        Validated and 1D-coerced ``(actual, predicted)`` arrays.

    Raises
    ------
    ValueError
        If arrays are not length-consistent or if ``actual`` is not binary.
    """
    actual = column_or_1d(actual)
    predicted = column_or_1d(predicted)
    check_consistent_length(actual, predicted)
    if type_of_target(actual, "actual") != "binary":
        raise ValueError("Expected 'actual' to be a binary target (e.g. 0/1 labels).")
    return actual, predicted


# %%
def check_metric_function(func):
    """
    Examples
    --------
    >>> def metric(a, b, *, sample_weight=None): return 1.0
    >>> check_metric_function(metric)
    True

    >>> def flexible_metric(a, b, **kwargs): return 1.0
    >>> check_metric_function(flexible_metric)
    True

    >>> def bad_metric(a, b): return 1.0
    >>> check_metric_function(bad_metric)
    False
    """
    sig = inspect.signature(func)
    try:
        sig.bind(object(), object(), sample_weight=None)
    except TypeError:
        return False
    return True


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
