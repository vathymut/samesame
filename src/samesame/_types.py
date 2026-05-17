"""Result types shared across samesame test modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from samesame._internals import Direction


@dataclass(frozen=True)
class TestResult:
    """Shared fields for all statistical test results."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftResult(TestResult):
    """Result of generic shift detection."""

    statistic_name: str
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class HarmResult(TestResult):
    """Result of harmful-shift detection."""

    direction: Direction
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class HarmInference:
    """Bayesian evidence layer for harmful shift."""

    posterior: NDArray[np.float64]
    bayes_factor: float


__all__ = ["HarmInference", "HarmResult", "ShiftResult", "TestResult"]
