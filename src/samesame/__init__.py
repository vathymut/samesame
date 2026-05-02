"""Task-first API for hypothesis tests over outlier scores.

The primary API exposes two functions:

- ``test_shift`` — test whether two outlier score distributions differ
- ``test_adverse_shift`` — test for harmful shifts with explicit direction

Both return a full result including the null distribution.
"""

from . import weights
from ._api import (
    AdverseShiftDetails,
    ShiftDetails,
    test_adverse_shift,
    test_shift,
)
from ._types import TestResult

__all__ = [
    "AdverseShiftDetails",
    "ShiftDetails",
    "TestResult",
    "test_adverse_shift",
    "test_shift",
    "weights",
]
