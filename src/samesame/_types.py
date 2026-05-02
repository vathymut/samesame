from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from samesame._utils import Direction

ShiftStatistic: TypeAlias = Literal[
    "roc_auc",
    "balanced_accuracy",
    "matthews_corrcoef",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Shared fields for all test results."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftDetails(TestResult):
    """Result of a shift test, including the full null distribution."""

    statistic_name: str
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class AdverseShiftDetails(TestResult):
    """Result of an adverse-shift test, including the full null distribution."""

    direction: Direction
    null_distribution: NDArray[np.float64]
    bayes_factor: float | None = None
    posterior: NDArray[np.float64] | None = None


__all__ = [
    "AdverseShiftDetails",
    "ShiftDetails",
    "ShiftStatistic",
    "TestResult",
]
