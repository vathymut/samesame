from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from samesame._utils import Direction
from samesame._weighting import NoWeighting, WeightingStrategy

ShiftStatistic: TypeAlias = Literal[
    "roc_auc",
    "balanced_accuracy",
    "matthews_corrcoef",
]


# ---------------------------------------------------------------------------
# Options dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResamplingOptions:
    """Shared resampling controls.

    Parameters
    ----------
    n_resamples : int, optional
        Number of permutation resamples, by default ``9999``.
    batch : int or None, optional
        Number of resamples to process per batch. ``None`` uses a single batch.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a fresh one.
    """

    n_resamples: int = 9999
    batch: int | None = None
    rng: np.random.Generator | None = None


@dataclass(frozen=True)
class ShiftOptions(ResamplingOptions):
    """Options for :func:`advanced.test_shift`.

    Parameters
    ----------
    weighting : WeightingStrategy, optional
        Weighting strategy. Defaults to :class:`NoWeighting`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis for the permutation test, by default ``'two-sided'``.
    """

    weighting: WeightingStrategy = field(default_factory=NoWeighting)
    alternative: Literal["less", "greater", "two-sided"] = "two-sided"


@dataclass(frozen=True)
class AdverseShiftOptions(ResamplingOptions):
    """Options for :func:`advanced.test_adverse_shift`.

    Parameters
    ----------
    weighting : WeightingStrategy, optional
        Weighting strategy. Defaults to :class:`NoWeighting`.
    bayesian : bool, optional
        When ``True``, also compute the Bayes factor and posterior draws,
        by default ``False``.
    """

    weighting: WeightingStrategy = field(default_factory=NoWeighting)
    bayesian: bool = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Shared summary fields for public test results."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftResult(TestResult):
    """Summary result for the primary shift test."""

    statistic_name: str


@dataclass(frozen=True)
class AdverseShiftResult(TestResult):
    """Summary result for the primary adverse-shift test."""

    direction: Direction


@dataclass(frozen=True)
class ShiftDetails(ShiftResult):
    """Detailed result for the advanced shift test."""

    null_distribution: NDArray[np.float64]

    def summary(self) -> ShiftResult:
        """Return the primary summary view of this detailed result."""
        return ShiftResult(
            statistic=self.statistic,
            pvalue=self.pvalue,
            statistic_name=self.statistic_name,
        )


@dataclass(frozen=True)
class AdverseShiftDetails(AdverseShiftResult):
    """Detailed result for the advanced adverse-shift test."""

    null_distribution: NDArray[np.float64]
    bayes_factor: float | None = None
    posterior: NDArray[np.float64] | None = None

    def summary(self) -> AdverseShiftResult:
        """Return the primary summary view of this detailed result."""
        return AdverseShiftResult(
            statistic=self.statistic,
            pvalue=self.pvalue,
            direction=self.direction,
        )


__all__ = [
    "AdverseShiftDetails",
    "AdverseShiftOptions",
    "AdverseShiftResult",
    "ResamplingOptions",
    "ShiftDetails",
    "ShiftOptions",
    "ShiftResult",
    "ShiftStatistic",
    "TestResult",
]
