"""Public shift-detection seam."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.multiclass import type_of_target

from samesame._internals import (
    Direction,
    RandomState,
    _bayes_factor,
    bayesian_posterior,
    build_two_sample_dataset,
    get_shift_statistic,
    requires_binary_scores,
    resolve_random_state,
    run_permutation_test,
    validate_and_normalise_weights,
    validate_direction,
    wauc,
)
from samesame._types import HarmInference, HarmResult, ShiftResult, TestResult
from samesame.weights import ImportanceWeights

type ShiftStatistic = Literal["roc_auc", "balanced_accuracy", "matthews_corrcoef"]


def _combine_importance_weights(
    weights: ImportanceWeights | None,
    *,
    n_source: int,
    n_target: int,
) -> NDArray | None:
    if weights is None:
        return None
    source_w = validate_and_normalise_weights(
        np.asarray(weights.source, dtype=float), n_source
    )
    target_w = validate_and_normalise_weights(
        np.asarray(weights.target, dtype=float), n_target
    )
    return np.concatenate([source_w, target_w])


def _validate_shift_scores(statistic_name: str, predicted: NDArray) -> None:
    if not requires_binary_scores(statistic_name):
        return
    if type_of_target(predicted, "predicted") != "binary":
        raise ValueError(
            f"statistic={statistic_name!r} requires binary outlier scores."
        )


def _prepare_harm_detection(
    source: ArrayLike,
    target: ArrayLike,
    direction: Direction,
    weights: ImportanceWeights | None,
) -> tuple[NDArray[np.int_], NDArray, Direction, NDArray | None]:
    dataset = build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    combined_weights = _combine_importance_weights(
        weights,
        n_source=dataset.n_source,
        n_target=dataset.n_target,
    )
    return actual, predicted, validated_direction, combined_weights


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
    dataset = build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    statistic_name, metric = get_shift_statistic(statistic)
    _validate_shift_scores(statistic_name, predicted)
    combined_weights = _combine_importance_weights(
        weights,
        n_source=dataset.n_source,
        n_target=dataset.n_target,
    )
    result = run_permutation_test(
        actual,
        predicted,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        sample_weight=combined_weights,
        rng=resolve_random_state(random_state),
        batch=batch,
    )
    return ShiftResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        statistic_name=statistic_name,
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
    actual, predicted, validated_direction, combined_weights = _prepare_harm_detection(
        source, target, direction, weights
    )
    result = run_permutation_test(
        actual,
        predicted,
        wauc,
        n_resamples=n_resamples,
        alternative="greater",
        sample_weight=combined_weights,
        rng=resolve_random_state(random_state),
        batch=batch,
    )
    return HarmResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
    )


def infer_harm(
    source: ArrayLike,
    target: ArrayLike,
    *,
    direction: Direction,
    n_resamples: int = 9999,
    random_state: RandomState = None,
    weights: ImportanceWeights | None = None,
    threshold: float = 1 / 12,
) -> HarmInference:
    """Infer Bayesian evidence for harmful shift."""
    actual, predicted, _direction, combined_weights = _prepare_harm_detection(
        source, target, direction, weights
    )
    posterior = np.asarray(
        bayesian_posterior(
            actual,
            predicted,
            wauc,
            n_resamples=n_resamples,
            rng=resolve_random_state(random_state),
            base_weight=combined_weights,
        ),
        dtype=np.float64,
    )
    return HarmInference(
        posterior=posterior,
        bayes_factor=float(_bayes_factor(posterior, threshold)),
    )


__all__ = [
    "Direction",
    "HarmInference",
    "HarmResult",
    "ShiftResult",
    "ShiftStatistic",
    "TestResult",
    "detect_harm",
    "detect_shift",
    "infer_harm",
]
