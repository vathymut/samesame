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


# %%
def check_metric_function(func):
    """
    Examples
    --------
    >>> def metric(a, b, *, sample_weight=None): return 1.0
    >>> check_metric_function(metric)
    True

    >>> def bad_metric(a, b): return 1.0
    >>> check_metric_function(bad_metric)
    False
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    # Two positional (or positional-or-keyword) arguments
    pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # Keyword-only argument named 'sample_weight'
    sample_weight_exists = any(
        p for p in params if p.kind == p.KEYWORD_ONLY and p.name == "sample_weight"
    )
    return len(pos) == 2 and sample_weight_exists


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
