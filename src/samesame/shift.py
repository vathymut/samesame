"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score

from samesame._comparison import (
    RandomNumberGenerator,
    prepare_two_sample_test,
    run_permutation_test,
)
from samesame._statistics import harmful_shift_statistic
from samesame.weights import ImportanceWeights


class Direction(Enum):
    """Polarity that defines "worse" for the scores.

    Attributes
    ----------
    HIGHER_IS_WORSE : Direction
        Larger scores indicate harm (e.g., predicted risk).
    HIGHER_IS_BETTER : Direction
        Larger scores indicate quality (e.g., confidence, accuracy).
    """

    HIGHER_IS_WORSE = "higher-is-worse"
    HIGHER_IS_BETTER = "higher-is-better"


RandomState = int | np.random.RandomState | np.random.Generator | None


@dataclass(frozen=True)
class TestResult:
    """Shared fields for all statistical test results.

    Attributes
    ----------
    statistic : float
        Observed value of the test statistic.
    pvalue : float
        Permutation p-value.
    null_distribution : NDArray[np.float64]
        Permutation null distribution of the statistic.
    """

    statistic: float
    pvalue: float
    null_distribution: NDArray[np.float64]

    def significant(self, alpha: float = 0.05) -> bool:
        """Return whether ``pvalue`` is significant at level ``alpha``.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Default is 0.05.

        Returns
        -------
        bool
            True when ``pvalue <= alpha``.

        Raises
        ------
        ValueError
            If ``alpha`` is not in the open interval (0, 1).
        """
        alpha_value = float(alpha)
        if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha must be in the open interval (0, 1).")
        return self.pvalue <= alpha_value

    def __repr__(self) -> str:
        rendered = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in fields(self)
            if field.name != "null_distribution"
        )
        return f"{type(self).__name__}({rendered})"


@dataclass(frozen=True, repr=False)
class ShiftResult(TestResult):
    """Result of generic shift detection."""


@dataclass(frozen=True, repr=False)
class HarmResult(TestResult):
    """Result of harmful-shift detection."""

    direction: Direction


def _validate_direction(direction: Direction) -> Direction:
    if not isinstance(direction, Direction):
        raise TypeError(
            f"direction must be a samesame.shift.Direction member; got {direction!r}."
        )
    return direction


def _resolve_random_state(random_state: RandomState) -> RandomNumberGenerator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator | np.random.RandomState):
        return random_state
    if isinstance(random_state, Integral):
        return np.random.default_rng(int(random_state))
    raise TypeError(
        "random_state must be an int, numpy.random.Generator, "
        "numpy.random.RandomState, or None."
    )


def _scores_for_direction(scores: NDArray, direction: Direction) -> NDArray:
    if direction is Direction.HIGHER_IS_BETTER:
        return -scores
    return scores


def _run_shift_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    metric: Callable[..., float],
    transform: Callable[[NDArray], NDArray] | None,
    alternative: Literal["less", "greater", "two-sided"],
    n_resamples: int,
    random_state: RandomState,
    weights: ImportanceWeights | None,
) -> TestResult:
    """Prepare arrays, resolve the RNG, and run the permutation test."""
    prepared = prepare_two_sample_test(source, target, weights=weights)
    scores = prepared.scores if transform is None else transform(prepared.scores)
    rng = _resolve_random_state(random_state)
    statistic, pvalue, null_distribution = run_permutation_test(
        prepared.labels,
        scores,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        rng=rng,
        sample_weight=prepared.sample_weight,
    )
    return TestResult(
        statistic=statistic,
        pvalue=pvalue,
        null_distribution=null_distribution,
    )


def _run_harm_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int,
    random_state: RandomState,
    weights: ImportanceWeights | None,
) -> HarmResult:
    """Prepare arrays and run the harmful-shift permutation test."""
    validated_direction = _validate_direction(direction)
    result = _run_shift_test(
        source,
        target,
        metric=harmful_shift_statistic,
        transform=lambda values: _scores_for_direction(values, validated_direction),
        alternative="greater",
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
    )
    return HarmResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        direction=validated_direction,
        null_distribution=result.null_distribution,
    )


def detect_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    n_resamples: int = 9999,
    random_state: RandomState = None,
    weights: ImportanceWeights | None = None,
) -> ShiftResult:
    """Detect whether Source and Target Outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    random_state : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    ShiftResult
        Observed statistic, p-value, and null distribution.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not positive.
    """
    result = _run_shift_test(
        source,
        target,
        metric=roc_auc_score,
        transform=None,
        alternative="two-sided",
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
    )
    return ShiftResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        null_distribution=result.null_distribution,
    )


def detect_harm(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int = 9999,
    random_state: RandomState = None,
    weights: ImportanceWeights | None = None,
) -> HarmResult:
    """Detect whether Target is harmfully shifted relative to Source.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
    direction : Direction
        Polarity that defines "worse" for the scores.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    random_state : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    HarmResult
        Observed statistic, p-value, direction, and null distribution.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not positive.
    TypeError
        If ``direction`` is not a ``Direction`` member.
    """
    return _run_harm_test(
        source,
        target,
        direction=direction,
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
    )


__all__ = [
    "Direction",
    "HarmResult",
    "ShiftResult",
    "TestResult",
    "detect_harm",
    "detect_shift",
]
