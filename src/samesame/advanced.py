"""Advanced expert-oriented API for score-vector hypothesis tests."""

from samesame._api import (
    AdverseShiftDetails,
    AdverseShiftOptions,
    ContextualRIWWeighting,
    NoWeighting,
    SampleWeighting,
    ShiftDetails,
    ShiftOptions,
    _advanced_test_adverse_shift as test_adverse_shift,
    _advanced_test_shift as test_shift,
)

__all__ = [
    # detailed result types
    "AdverseShiftDetails",
    "ShiftDetails",
    # options types
    "AdverseShiftOptions",
    "ShiftOptions",
    # weighting strategies
    "ContextualRIWWeighting",
    "NoWeighting",
    "SampleWeighting",
    # functions
    "test_adverse_shift",
    "test_shift",
]
