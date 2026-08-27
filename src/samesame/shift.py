"""Shift tests: does the target differ, and is the difference harmful?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score

from samesame._permutation import Seed, _permutation_test
from samesame._statistics import harmful_shift_statistic
from samesame.weights import ImportanceWeights

Worse = Literal["higher", "lower"]


def _fmt(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float) else repr(v)


@dataclass(frozen=True)
class ShiftResult:
    """Result of :func:`test_shift`.

    Attributes
    ----------
    statistic : float
        Observed test statistic (ROC AUC).
    pvalue : float
        Permutation p-value (two-sided).
    null_distribution : NDArray[np.float64]
        Null distribution of the statistic.
    """

    statistic: float
    pvalue: float
    null_distribution: NDArray[np.float64]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"statistic={_fmt(self.statistic)}, pvalue={_fmt(self.pvalue)})"
        )


@dataclass(frozen=True, repr=False)
class HarmfulShiftResult(ShiftResult):
    """Result of :func:`test_harmful_shift`.

    Attributes
    ----------
    statistic : float
        Observed harmful-shift statistic.
    pvalue : float
        Permutation p-value (one-sided, greater).
    null_distribution : NDArray[np.float64]
        Null distribution of the statistic.
    worse : {'higher', 'lower'}
        Declared harmful direction.
    """

    worse: Worse  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"statistic={_fmt(self.statistic)}, pvalue={_fmt(self.pvalue)}, "
            f"worse={self.worse!r})"
        )


# ---------------------------------------------------------------------------
# internal metrics
# ---------------------------------------------------------------------------


def _auc_metric(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
) -> float:
    return float(roc_auc_score(labels, scores, sample_weight=sample_weight))


def _harm_metric_factory(worse: Worse):
    def _metric(
        labels: NDArray[np.int_],
        scores: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None,
    ) -> float:
        polarity = scores if worse == "higher" else -scores
        return harmful_shift_statistic(labels, polarity, sample_weight=sample_weight)

    return _metric


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def test_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    n_resamples: int = 9999,
    rng: Seed = None,
    weights: ImportanceWeights | None = None,
) -> ShiftResult:
    """Test whether source and target score distributions differ.

    Two-sided permutation test using ROC AUC. A small p-value is evidence
    that the groups differ.

    Parameters
    ----------
    source : ArrayLike
        Scores from the source (reference) group. Generate out-of-sample
        when they come from a fitted model (cross-validation, OOB, or
        held-out set); in-sample predictions can invalidate the test.
    target : ArrayLike
        Scores from the target (evaluation) group.
    n_resamples : int, optional
        Number of permutation resamples. Default 9999.
    rng : int | np.random.Generator | np.random.RandomState | None, optional
        Random state for reproducibility. Pass an ``int`` seed or a
        ``Generator``/``RandomState``. Default ``None``.
    weights : ImportanceWeights | None, optional
        Importance weights per group.

    Returns
    -------
    ShiftResult
        Observed statistic, p-value, and null distribution.
    """
    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=_auc_metric,
        alternative="two-sided",
        n_resamples=n_resamples,
        rng=rng,
        weights=weights,
    )
    return ShiftResult(
        statistic=statistic, pvalue=pvalue, null_distribution=null_distribution
    )


def test_harmful_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    worse: Worse,
    n_resamples: int = 9999,
    rng: Seed = None,
    weights: ImportanceWeights | None = None,
) -> HarmfulShiftResult:
    """Test whether target is harmfully shifted relative to source.

    One-sided permutation test. A small p-value is evidence that target
    has excess mass in the harmful tail.

    Parameters
    ----------
    source : ArrayLike
        Scores from the source (reference) group.
    target : ArrayLike
        Scores from the target (evaluation) group.
    worse : {'higher', 'lower'}
        Whether larger (``'higher'``) or smaller (``'lower'``) scores
        indicate harm.
    n_resamples : int, optional
        Number of permutation resamples. Default 9999.
    rng : int | np.random.Generator | np.random.RandomState | None, optional
        Random state for reproducibility.
    weights : ImportanceWeights | None, optional
        Importance weights per group.

    Returns
    -------
    HarmfulShiftResult
        Observed statistic, p-value, harm direction, and null distribution.
    """
    if worse not in ("higher", "lower"):
        raise ValueError("worse must be either 'higher' or 'lower'.")

    metric = _harm_metric_factory(worse)

    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=metric,
        alternative="greater",
        n_resamples=n_resamples,
        rng=rng,
        weights=weights,
    )
    return HarmfulShiftResult(
        statistic=statistic,
        pvalue=pvalue,
        worse=worse,
        null_distribution=null_distribution,
    )


__all__ = ["HarmfulShiftResult", "ShiftResult", "test_harmful_shift", "test_shift"]
