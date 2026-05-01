from __future__ import annotations

from numpy.typing import ArrayLike

from samesame._types import (
    AdverseShiftDetails,
    AdverseShiftOptions,
    AdverseShiftResult,
    ResamplingOptions,
    ShiftDetails,
    ShiftOptions,
    ShiftResult,
    ShiftStatistic,
    TestResult,
)
from samesame._utils import Direction
from samesame._weighting import (
    ContextualRIWWeighting,
    NoWeighting,
    SampleWeighting,
    WeightingStrategy,
)
from samesame.advanced import (
    test_adverse_shift as _adv_test_adverse_shift,
    test_shift as _adv_test_shift,
)


# ---------------------------------------------------------------------------
# Primary (simple) API
# ---------------------------------------------------------------------------


def test_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    options: ShiftOptions = ShiftOptions(),
) -> ShiftResult:
    """Test whether two outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
    statistic : {'roc_auc', 'balanced_accuracy', 'matthews_corrcoef'}, optional
        Named built-in statistic used inside the permutation test.
    options : ShiftOptions, optional
        Resampling, weighting, and alternative-hypothesis controls.

    Returns
    -------
    ShiftResult
        Immutable summary containing the test statistic and p-value.
    """
    return _adv_test_shift(
        source=source,
        target=target,
        statistic=statistic,
        options=options,
    ).summary()


def test_adverse_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    direction: Direction,
    options: AdverseShiftOptions = AdverseShiftOptions(),
) -> AdverseShiftResult:
    """Test whether the target sample is harmfully shifted.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
    direction : {'higher-is-worse', 'higher-is-better'}
        Semantic direction of larger score values.
    options : AdverseShiftOptions, optional
        Resampling, weighting, and Bayesian controls.

    Returns
    -------
    AdverseShiftResult
        Immutable summary containing the test statistic and p-value.
    """
    return _adv_test_adverse_shift(
        source=source,
        target=target,
        direction=direction,
        options=options,
    ).summary()


__all__ = [
    # result types
    "AdverseShiftDetails",
    "AdverseShiftResult",
    "ShiftDetails",
    "ShiftResult",
    "TestResult",
    # options types
    "AdverseShiftOptions",
    "ResamplingOptions",
    "ShiftOptions",
    # weighting strategies
    "ContextualRIWWeighting",
    "NoWeighting",
    "SampleWeighting",
    "WeightingStrategy",
    # public functions
    "test_adverse_shift",
    "test_shift",
]
