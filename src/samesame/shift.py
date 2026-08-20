"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
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
ShiftStatistic = Literal["roc_auc", "balanced_accuracy", "matthews_corrcoef"]


@dataclass(frozen=True, repr=False)
class ShiftResult(TestResult):
    """Result of generic shift detection."""

    statistic_name: str


@dataclass(frozen=True, repr=False)
class HarmResult(TestResult):
    """Result of harmful-shift detection."""

    direction: Direction


@dataclass(frozen=True, repr=False)
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


def _validate_direction(direction: Direction) -> Direction:
    if not isinstance(direction, Direction):
        raise TypeError(
            "direction must be a samesame.shift.Direction member; "
            f"got {direction!r}."
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
    validate: Callable[[PreparedTwoSampleTest], None] | None = None,
) -> tuple[PreparedTwoSampleTest, NDArray, TestResult, RandomNumberGenerator]:
    """Prepare arrays, resolve the RNG, and run the permutation test."""
    prepared = prepare_two_sample_test(source, target, weights=weights)
    if validate is not None:
        validate(prepared)
    scores = prepared.scores if transform is None else transform(prepared.scores)
    rng = _resolve_random_state(random_state)
    result = run_permutation_test(
        prepared.labels,
        scores,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        rng=rng,
        sample_weight=prepared.sample_weight,
    )
    return prepared, scores, result, rng


def _run_harm_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int,
    random_state: RandomState,
    weights: ImportanceWeights | None,
) -> _PreparedHarmTest:
    """Prepare arrays and run the harmful-shift permutation test."""
    validated_direction = _validate_direction(direction)
    prepared, scores, result, rng = _run_shift_test(
        source,
        target,
        metric=harmful_shift_statistic,
        transform=lambda values: _scores_for_direction(values, validated_direction),
        alternative="greater",
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
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
        If ``n_resamples`` is not positive.
    ValueError
        If ``statistic`` is not one of the supported statistics.
    """
    metric = _get_shift_statistic(statistic)
    _, _, result, _ = _run_shift_test(
        source,
        target,
        metric=metric,
        transform=None,
        alternative=alternative,
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
        validate=lambda p: _validate_shift_scores(statistic, p.scores),
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
    bayesian: bool = False,
    threshold: float | None = None,
    n_resamples: int = 9999,
    random_state: RandomState = None,
    weights: ImportanceWeights | None = None,
) -> HarmResult | BayesianHarmResult:
    """Detect whether Target is harmfully shifted relative to Source.

    Parameters
    ----------
    source : ArrayLike
        Outlier scores from the source (reference) group.
    target : ArrayLike
        Outlier scores from the target (evaluation) group.
    direction : Direction
        Polarity that defines "worse" for the scores.
    bayesian : bool, optional
        When True, also compute posterior evidence for harmful shift
        (posterior draws and Bayes factor). Default is False.
    threshold : float | None, optional
        Statistic value above which a posterior draw counts as evidence of
        harm. Only meaningful when ``bayesian=True``. Default ``1 / 12``.
    n_resamples : int, optional
        Number of permutation resamples. Default is 9999.
    random_state : int | np.random.RandomState | np.random.Generator | None, optional
        Random seed or generator for the permutation test.
    weights : ImportanceWeights | None, optional
        Importance weights for the source and target groups.

    Returns
    -------
    HarmResult or BayesianHarmResult
        Observed statistic, p-value, direction, and null distribution. With
        ``bayesian=True``, also posterior draws and Bayes factor.

    Raises
    ------
    ValueError
        If ``n_resamples`` is not positive.
    ValueError
        If ``threshold`` is not finite, or is provided without
        ``bayesian=True``.
    TypeError
        If ``direction`` is not a ``Direction`` member.
    """
    if threshold is not None and not bayesian:
        raise ValueError("threshold is only meaningful when bayesian=True.")
    prepared_test = _run_harm_test(
        source,
        target,
        direction=direction,
        n_resamples=n_resamples,
        random_state=random_state,
        weights=weights,
    )
    harm_result = HarmResult(
        statistic=prepared_test.result.statistic,
        pvalue=prepared_test.result.pvalue,
        direction=prepared_test.direction,
        null_distribution=prepared_test.result.null_distribution,
    )
    if not bayesian:
        return harm_result
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
        statistic=harm_result.statistic,
        pvalue=harm_result.pvalue,
        direction=harm_result.direction,
        null_distribution=harm_result.null_distribution,
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
    "detect_shift",
]
