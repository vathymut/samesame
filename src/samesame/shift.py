"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target

from samesame._comparison import (
    PreparedTwoSampleTest,
    RandomNumberGenerator,
    TestResult,
    prepare_two_sample_test,
    run_permutation_test,
)
from samesame._posterior import compute_posterior_evidence
from samesame._statistics import harmful_shift_statistic
from samesame.weights import ImportanceWeights

Direction = Literal["higher-is-worse", "higher-is-better"]
RandomState = int | np.random.RandomState | np.random.Generator | None
ShiftStatistic = Literal["roc_auc", "balanced_accuracy", "matthews_corrcoef"]


@dataclass(frozen=True)
class ShiftResult(TestResult):
    """Result of generic shift detection."""

    statistic_name: str


@dataclass(frozen=True)
class HarmResult(TestResult):
    """Result of harmful-shift detection."""

    direction: Direction


@dataclass(frozen=True)
class BayesianHarmResult(HarmResult):
    """Result of Bayesian harmful-shift detection."""

    posterior: NDArray[np.float64]
    bayes_factor: float


@dataclass(frozen=True)
class _PreparedHarmTest:
    """Prepared inputs and raw result for harmful-shift tests."""

    prepared: PreparedTwoSampleTest
    scores: NDArray
    direction: Direction
    result: TestResult
    rng: RandomNumberGenerator


_SHIFT_STATISTICS: dict[str, Callable[..., float]] = {
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "matthews_corrcoef": matthews_corrcoef,
}


def _validate_direction(direction: str) -> Direction:
    match direction:
        case "higher-is-worse" | "higher-is-better":
            return direction
    raise ValueError(
        "direction must be one of 'higher-is-worse' or 'higher-is-better'."
    )


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


def _get_shift_statistic(name: str) -> Callable[..., float]:
    statistic = _SHIFT_STATISTICS.get(name)
    if statistic is None:
        allowed = ", ".join(sorted(_SHIFT_STATISTICS))
        raise ValueError(f"statistic must be one of {allowed}; got {name!r}.")
    return statistic


def _resolve_posterior_threshold(threshold: float | None) -> float:
    if threshold is None:
        return 1 / 12
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    return threshold_value


def _validate_shift_scores(statistic_name: str, predicted: NDArray) -> None:
    if (
        statistic_name in {"balanced_accuracy", "matthews_corrcoef"}
        and type_of_target(predicted, "predicted") != "binary"
    ):
        raise ValueError(
            f"statistic={statistic_name!r} requires binary outlier scores."
        )


def _scores_for_direction(scores: NDArray, direction: Direction) -> NDArray:
    if direction == "higher-is-better":
        return -scores
    return scores


def _run_harm_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int,
    batch: int | None,
    random_state: RandomState,
    weights: ImportanceWeights | None,
) -> _PreparedHarmTest:
    """Prepare arrays and run the harmful-shift permutation test."""
    prepared = prepare_two_sample_test(source, target, weights=weights)
    validated_direction = _validate_direction(direction)
    scores = _scores_for_direction(prepared.scores, validated_direction)
    rng = _resolve_random_state(random_state)
    result = run_permutation_test(
        prepared.labels,
        scores,
        harmful_shift_statistic,
        n_resamples=n_resamples,
        batch=batch,
        alternative="greater",
        rng=rng,
        sample_weight=prepared.sample_weight,
    )
    return _PreparedHarmTest(
        prepared=prepared,
        scores=scores,
        direction=validated_direction,
        result=result,
        rng=rng,
    )


def detect_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    statistic: ShiftStatistic = "roc_auc",
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_resamples: int = 9999,
    batch: int | None = None,
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
    statistic : {'roc_auc', 'balanced_accuracy', 'matthews_corrcoef'}, optional
        Two-sample score statistic. Default is ``'roc_auc'``.
    alternative : {'less', 'greater', 'two-sided'}, optional
        Alternative hypothesis for the permutation test. Default is
        ``'two-sided'``.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    batch : int | None, optional
        Number of permutations to evaluate per batch, or ``None`` for no
        batching.
    random_state : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    ShiftResult
        Observed statistic, p-value, statistic name, and null distribution.

    Raises
    ------
    ValueError
        If ``n_resamples`` or ``batch`` is not positive.
    ValueError
        If ``statistic`` is not one of the supported statistics.
    """
    prepared = prepare_two_sample_test(source, target, weights=weights)
    metric = _get_shift_statistic(statistic)
    _validate_shift_scores(statistic, prepared.scores)
    result = run_permutation_test(
        prepared.labels,
        prepared.scores,
        metric,
        n_resamples=n_resamples,
        batch=batch,
        alternative=alternative,
        rng=_resolve_random_state(random_state),
        sample_weight=prepared.sample_weight,
    )
    return ShiftResult(
        statistic=result.statistic,
        pvalue=result.pvalue,
        statistic_name=statistic,
        null_distribution=result.null_distribution,
    )


def detect_harm(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int = 9999,
    batch: int | None = None,
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
    direction : {'higher-is-worse', 'higher-is-better'}
        Polarity that defines "worse" for the scores.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    batch : int | None, optional
        Number of permutations to evaluate per batch, or ``None`` for no
        batching.
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
        If ``n_resamples`` or ``batch`` is not positive.
    ValueError
        If ``direction`` is not one of the supported directions.
    """
    prepared_test = _run_harm_test(
        source,
        target,
        direction=direction,
        n_resamples=n_resamples,
        batch=batch,
        random_state=random_state,
        weights=weights,
    )
    return HarmResult(
        statistic=prepared_test.result.statistic,
        pvalue=prepared_test.result.pvalue,
        direction=prepared_test.direction,
        null_distribution=prepared_test.result.null_distribution,
    )


def detect_harm_bayesian(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int = 9999,
    batch: int | None = None,
    random_state: RandomState = None,
    weights: ImportanceWeights | None = None,
    threshold: float | None = None,
) -> BayesianHarmResult:
    """Detect harmful shift and compute Bayesian posterior evidence.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
    direction : {'higher-is-worse', 'higher-is-better'}
        Polarity that defines "worse" for the scores.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    batch : int | None, optional
        Number of permutations to evaluate per batch, or ``None`` for no
        batching.
    random_state : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.
    threshold : float | None, optional
        Threshold above which the harmful-shift statistic counts as evidence
        of harm. Default ``1 / 12``.

    Returns
    -------
    BayesianHarmResult
        Observed statistic, p-value, direction, null distribution, posterior
        draws, and Bayes factor.

    Raises
    ------
    ValueError
        If ``n_resamples`` or ``batch`` is not positive.
    ValueError
        If ``direction`` is not one of the supported directions.
    ValueError
        If ``threshold`` is not finite.
    """
    prepared_test = _run_harm_test(
        source,
        target,
        direction=direction,
        n_resamples=n_resamples,
        batch=batch,
        random_state=random_state,
        weights=weights,
    )
    resolved_threshold = _resolve_posterior_threshold(threshold)
    posterior, bayes_factor = compute_posterior_evidence(
        prepared_test.prepared.labels,
        prepared_test.scores,
        harmful_shift_statistic,
        threshold=resolved_threshold,
        n_resamples=n_resamples,
        rng=prepared_test.rng,
        base_weight=prepared_test.prepared.sample_weight,
    )
    return BayesianHarmResult(
        statistic=prepared_test.result.statistic,
        pvalue=prepared_test.result.pvalue,
        direction=prepared_test.direction,
        null_distribution=prepared_test.result.null_distribution,
        posterior=posterior,
        bayes_factor=bayes_factor,
    )


__all__ = [
    "BayesianHarmResult",
    "Direction",
    "HarmResult",
    "ShiftResult",
    "ShiftStatistic",
    "TestResult",
    "detect_harm",
    "detect_harm_bayesian",
    "detect_shift",
]
