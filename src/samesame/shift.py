"""Public shift-detection seam."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score

from samesame._permutation import _permutation_test
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


def _resolve_rng(rng: Rng) -> np.random.Generator | np.random.RandomState:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator | np.random.RandomState):
        return rng
    raise TypeError(
        "rng must be a numpy.random.Generator, numpy.random.RandomState, or None."
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
    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=roc_auc_score,
        alternative="two-sided",
        n_resamples=n_resamples,
        rng=_resolve_rng(rng),
        weights=weights,
    )
    return ShiftResult(
        statistic=statistic,
        pvalue=pvalue,
        null_distribution=null_distribution,
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
    if worse not in ("higher", "lower"):
        raise ValueError("worse must be either 'higher' or 'lower'.")

    def metric(
        labels: NDArray[np.int_],
        scores: NDArray,
        sample_weight: NDArray[np.float64] | None,
    ) -> float:
        polarity = scores if worse == "higher" else -scores
        return harmful_shift_statistic(labels, polarity, sample_weight=sample_weight)

    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=metric,
        alternative="greater",
        n_resamples=n_resamples,
        rng=_resolve_rng(rng),
        weights=weights,
    )
    return HarmfulShiftResult(
        statistic=statistic,
        pvalue=pvalue,
        worse=worse,
        null_distribution=null_distribution,
    )


__all__ = [
    "HarmfulShiftResult",
    "ShiftResult",
    "test_harmful_shift",
    "test_shift",
]
