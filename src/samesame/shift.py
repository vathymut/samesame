"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
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

Rng = int | np.random.RandomState | np.random.Generator | None
Worse = Literal["higher", "lower"]


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

    worse: Worse


def _validate_worse(worse: str) -> Worse:
    if worse not in ("higher", "lower"):
        raise ValueError("worse must be either 'higher' or 'lower'.")
    return worse  # type: ignore[return-value]


def _resolve_rng(rng: Rng) -> RandomNumberGenerator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator | np.random.RandomState):
        return rng
    if isinstance(rng, Integral):
        return np.random.default_rng(int(rng))
    raise TypeError(
        "rng must be an int, numpy.random.Generator, "
        "numpy.random.RandomState, or None."
    )


def _scores_for_worse(scores: NDArray, worse: Worse) -> NDArray:
    if worse == "lower":
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
    batch: int | None,
    rng: Rng,
    weights: ImportanceWeights | None,
) -> TestResult:
    """Prepare arrays, resolve the RNG, and run the permutation test."""
    prepared = prepare_two_sample_test(source, target, weights=weights)
    scores = prepared.scores if transform is None else transform(prepared.scores)
    rng = _resolve_rng(rng)
    statistic, pvalue, null_distribution = run_permutation_test(
        prepared.labels,
        scores,
        metric,
        n_resamples=n_resamples,
        batch=batch,
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
    worse: Worse,
    n_resamples: int,
    batch: int | None,
    rng: Rng,
    weights: ImportanceWeights | None,
) -> HarmResult:
    """Prepare arrays and run the harmful-shift permutation test."""
    validated_worse = _validate_worse(worse)
    result = _run_shift_test(
        source,
        target,
        metric=harmful_shift_statistic,
        transform=lambda values: _scores_for_worse(values, validated_worse),
        alternative="greater",
        n_resamples=n_resamples,
        batch=batch,
        rng=rng,
        weights=weights,
    )
    return HarmResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        worse=validated_worse,
        null_distribution=result.null_distribution,
    )


def detect_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    n_resamples: int = 9999,
    batch: int | None = None,
    rng: Rng = None,
    weights: ImportanceWeights | None = None,
) -> ShiftResult:
    """Detect whether Source and Target Outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
        When scores come from a fitted model, generate them out of sample
        with cross-validation, out-of-bag predictions, or a held-out set.
        In-sample predictions can invalidate the test interpretation.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
        When scores come from a fitted model, generate them out of sample
        with cross-validation, out-of-bag predictions, or a held-out set.
        In-sample predictions can invalidate the test interpretation.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    batch : int or None, optional
        Number of permutations processed at once. Controls memory usage and
        runtime, not the number of resamples. Default is None.
    rng : int | np.random.RandomState | np.random.Generator | None, optional
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
        batch=batch,
        rng=rng,
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
    worse: Worse,
    n_resamples: int = 9999,
    batch: int | None = None,
    rng: Rng = None,
    weights: ImportanceWeights | None = None,
) -> HarmResult:
    """Detect whether Target is harmfully shifted relative to Source.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
        When scores come from a fitted model, generate them out of sample
        with cross-validation, out-of-bag predictions, or a held-out set.
        In-sample predictions can invalidate the test interpretation.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
        When scores come from a fitted model, generate them out of sample
        with cross-validation, out-of-bag predictions, or a held-out set.
        In-sample predictions can invalidate the test interpretation.
    worse : {'higher', 'lower'}
        Whether higher or lower scores indicate worse outcomes.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    batch : int or None, optional
        Number of permutations processed at once. Controls memory usage and
        runtime, not the number of resamples. Default is None.
    rng : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    HarmResult
        Observed statistic, p-value, worse direction, and null distribution.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not positive.
    ValueError
        If ``worse`` is not ``"higher"`` or ``"lower"``.
    """
    return _run_harm_test(
        source,
        target,
        worse=worse,
        n_resamples=n_resamples,
        batch=batch,
        rng=rng,
        weights=weights,
    )


__all__ = [
    "HarmResult",
    "ShiftResult",
    "TestResult",
    "Worse",
    "detect_harm",
    "detect_shift",
]
