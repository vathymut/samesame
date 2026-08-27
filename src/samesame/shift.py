"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
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

Rng = np.random.RandomState | np.random.Generator | None


@dataclass(frozen=True)
class _TestResult:
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

    def __repr__(self) -> str:
        rendered = ", ".join(
            f"{field.name}={getattr(self, field.name)!r}"
            for field in fields(self)
            if field.name != "null_distribution"
        )
        return f"{type(self).__name__}({rendered})"


@dataclass(frozen=True, repr=False)
class ShiftResult(_TestResult):
    """Result of generic shift detection."""


@dataclass(frozen=True, repr=False)
class HarmfulShiftResult(_TestResult):
    """Result of harmful-shift detection."""

    worse: Literal["higher", "lower"]


def _resolve_rng(rng: Rng) -> RandomNumberGenerator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator | np.random.RandomState):
        return rng
    raise TypeError(
        "rng must be a numpy.random.Generator, "
        "numpy.random.RandomState, or None."
    )


def _scores_for_worse(scores: NDArray, worse: Literal["higher", "lower"]) -> NDArray:
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
    rng: Rng,
    weights: ImportanceWeights | None,
) -> _TestResult:
    """Prepare arrays, resolve the RNG, and run the permutation test."""
    prepared = prepare_two_sample_test(source, target, weights=weights)
    scores = prepared.scores if transform is None else transform(prepared.scores)
    rng = _resolve_rng(rng)
    statistic, pvalue, null_distribution = run_permutation_test(
        prepared.labels,
        scores,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        rng=rng,
        sample_weight=prepared.sample_weight,
    )
    return _TestResult(
        statistic=statistic,
        pvalue=pvalue,
        null_distribution=null_distribution,
    )


def _run_harm_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    worse: Literal["higher", "lower"],
    n_resamples: int,
    rng: Rng,
    weights: ImportanceWeights | None,
) -> HarmfulShiftResult:
    """Prepare arrays and run the harmful-shift permutation test."""
    if worse not in ("higher", "lower"):
        raise ValueError("worse must be either 'higher' or 'lower'.")
    result = _run_shift_test(
        source,
        target,
        metric=harmful_shift_statistic,
        transform=lambda values: _scores_for_worse(values, worse),
        alternative="greater",
        n_resamples=n_resamples,
        rng=rng,
        weights=weights,
    )
    return HarmfulShiftResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        worse=worse,
        null_distribution=result.null_distribution,
    )


def test_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    n_resamples: int = 9999,
    rng: Rng = None,
    weights: ImportanceWeights | None = None,
) -> ShiftResult:
    """Test whether Source and Target Outlier score distributions differ.

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
    rng : np.random.RandomState | np.random.Generator | None, optional
        Random generator for the permutation test. Integer seeds are not
        accepted; use ``np.random.default_rng(seed)`` when reproducibility
        from a seed is needed.
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
        rng=rng,
        weights=weights,
    )
    return ShiftResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        null_distribution=result.null_distribution,
    )


def test_harmful_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    worse: Literal["higher", "lower"],
    n_resamples: int = 9999,
    rng: Rng = None,
    weights: ImportanceWeights | None = None,
) -> HarmfulShiftResult:
    """Test whether Target is harmfully shifted relative to Source.

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
    rng : np.random.RandomState | np.random.Generator | None, optional
        Random generator for the permutation test. Integer seeds are not
        accepted; use ``np.random.default_rng(seed)`` when reproducibility
        from a seed is needed.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    HarmfulShiftResult
        Observed statistic, p-value, and harm direction.

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
        rng=rng,
        weights=weights,
    )


__all__ = [
    "HarmfulShiftResult",
    "ShiftResult",
    "test_harmful_shift",
    "test_shift",
]
