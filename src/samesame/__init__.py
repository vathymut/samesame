"""Task-first API for hypothesis tests over outlier scores.

The primary API exposes:

- ``test_shift`` — test whether two outlier score distributions differ
- ``test_adverse_shift`` — test for harmful shifts with explicit direction
- ``adverse_shift_posterior`` — Bayesian evidence layer on top of an adverse-shift result

All test functions return a full result including the null distribution.
"""

from . import weights
from ._api import (
    AdverseShiftDetails,
    BayesianEvidence,
    ShiftDetails,
    adverse_shift_posterior,
    test_adverse_shift,
    test_shift,
)
from ._types import TestResult
from .weights import ContextualWeights

__all__ = [
    "AdverseShiftDetails",
    "BayesianEvidence",
    "ContextualWeights",
    "ShiftDetails",
    "TestResult",
    "adverse_shift_posterior",
    "test_adverse_shift",
    "test_shift",
    "weights",
]
