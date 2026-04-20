"""Task-first API for hypothesis tests over score vectors.

The primary API exposes two functions:

- ``test_shift`` for generic score-distribution differences
- ``test_adverse_shift`` for harmful score shifts with explicit direction
"""

from . import advanced
from ._api import (
    AdverseShiftResult,
    ShiftResult,
    test_adverse_shift,
    test_shift,
)

__all__ = [
    "AdverseShiftResult",
    "ShiftResult",
    "advanced",
    "test_adverse_shift",
    "test_shift",
]
