"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target

from samesame._comparison import prepare_two_sample_test
from samesame._posterior import compute_posterior_evidence
from samesame._statistics import harmful_shift_statistic
from samesame.weights import ImportanceWeights

Direction = Literal["higher-is-worse", "higher-is-better"]
RandomState = int | np.random.RandomState | np.random.Generator | None
ShiftStatistic = Literal["roc_auc", "balanced_accuracy", "matthews_corrcoef"]


@dataclass(frozen=True)
class ShiftResult:
    """Result of generic shift detection."""

    statistic: float
    pvalue: float
    statistic_name: str
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class HarmResult:
    """Result of harmful-shift detection."""

    statistic: float
    pvalue: float
    direction: Direction
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class BayesianHarmResult:
    """Result of Bayesian harmful-shift detection."""

    statistic: float
    pvalue: float
    direction: Direction
    null_distribution: NDArray[np.float64]
    posterior: NDArray[np.float64]
    bayes_factor: float


_SHIFT_STATISTICS: dict[str, Callable[..., float]] = {
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "matthews_corrcoef": matthews_corrcoef,
}


def _validate_direction(direction: str) -> Direction:
    if direction not in ("higher-is-worse", "higher-is-better"):
        raise ValueError(
            "direction must be one of 'higher-is-worse' or 'higher-is-better'."
        )
    return direction


def _resolve_random_state(
    random_state: RandomState,
) -> np.random.Generator | np.random.RandomState:
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


def _validate_permutation_params(n_resamples: int, batch: int | None) -> None:
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if batch is not None and batch < 1:
        raise ValueError("batch must be a positive integer or None.")


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
    """Detect whether Source and Target Outlier score distributions differ."""
    _validate_permutation_params(n_resamples, batch)
    prepared = prepare_two_sample_test(source, target, weights=weights)
    metric = _get_shift_statistic(statistic)
    _validate_shift_scores(statistic, prepared.scores)
    rng = _resolve_random_state(random_state)
    _perm_weights = (
        None
        if prepared.sample_weight is None
        else np.asarray(prepared.sample_weight, dtype=float)
    )

    def _statistic(labels: NDArray[np.int_], scores: NDArray) -> float:
        return float(metric(labels, scores, sample_weight=_perm_weights))

    result = permutation_test(
        data=(prepared.labels, prepared.scores),
        statistic=_statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        batch=batch,
        alternative=alternative,
        rng=rng,
    )
    return ShiftResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        statistic_name=statistic,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
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
    """Detect whether Target is harmfully shifted relative to Source."""
    _validate_permutation_params(n_resamples, batch)
    prepared = prepare_two_sample_test(source, target, weights=weights)
    validated_direction = _validate_direction(direction)
    scores = _scores_for_direction(prepared.scores, validated_direction)
    rng = _resolve_random_state(random_state)
    _perm_weights = (
        None
        if prepared.sample_weight is None
        else np.asarray(prepared.sample_weight, dtype=float)
    )

    def _statistic(labels: NDArray[np.int_], _scores: NDArray) -> float:
        return float(
            harmful_shift_statistic(labels, _scores, sample_weight=_perm_weights)
        )

    result = permutation_test(
        data=(prepared.labels, scores),
        statistic=_statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        batch=batch,
        alternative="greater",
        rng=rng,
    )
    return HarmResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
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
    """Detect harmful shift and compute Bayesian posterior evidence."""
    _validate_permutation_params(n_resamples, batch)
    prepared = prepare_two_sample_test(source, target, weights=weights)
    validated_direction = _validate_direction(direction)
    scores = _scores_for_direction(prepared.scores, validated_direction)
    resolved_threshold = _resolve_posterior_threshold(threshold)
    rng = _resolve_random_state(random_state)
    _perm_weights = (
        None
        if prepared.sample_weight is None
        else np.asarray(prepared.sample_weight, dtype=float)
    )

    def _statistic(labels: NDArray[np.int_], _scores: NDArray) -> float:
        return float(
            harmful_shift_statistic(labels, _scores, sample_weight=_perm_weights)
        )

    result = permutation_test(
        data=(prepared.labels, scores),
        statistic=_statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        batch=batch,
        alternative="greater",
        rng=rng,
    )
    posterior, bayes_factor = compute_posterior_evidence(
        prepared.labels,
        scores,
        harmful_shift_statistic,
        threshold=resolved_threshold,
        n_resamples=n_resamples,
        rng=rng,
        base_weight=prepared.sample_weight,
    )
    return BayesianHarmResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
        posterior=posterior,
        bayes_factor=bayes_factor,
    )


__all__ = [
    "BayesianHarmResult",
    "Direction",
    "HarmResult",
    "ShiftResult",
    "ShiftStatistic",
    "detect_harm",
    "detect_harm_bayesian",
    "detect_shift",
]
