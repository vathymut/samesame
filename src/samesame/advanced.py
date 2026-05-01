"""Advanced expert-oriented API for score-vector hypothesis tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils.multiclass import type_of_target

from samesame._bayesboot import bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._metrics import get_shift_metric, requires_binary_scores, wauc
from samesame._stats import _bayes_factor
from samesame._types import (
    AdverseShiftDetails,
    AdverseShiftOptions,
    ShiftDetails,
    ShiftOptions,
    ShiftStatistic,
)
from samesame._utils import (
    Direction,
    validate_and_normalise_weights,
    validate_direction,
)
from samesame._weighting import (
    ContextualRIWWeighting,
    NoWeighting,
    SampleWeighting,
    WeightingStrategy,
    _resolve_weighting,
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_shift_scores(statistic_name: str, predicted: NDArray) -> None:
    if not requires_binary_scores(statistic_name):
        return
    if type_of_target(predicted, "predicted") != "binary":
        raise ValueError(
            f"statistic={statistic_name!r} requires binary outlier scores."
        )


def _prepare_adverse_inputs(
    *,
    source: ArrayLike,
    target: ArrayLike,
    direction: str,
) -> tuple[NDArray[np.int_], NDArray, Direction]:
    actual, predicted = build_two_sample_dataset(source, target)
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    return actual, predicted, validated_direction


def _run_permutation_test(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable,
    *,
    n_resamples: int,
    alternative: Literal["less", "greater", "two-sided"],
    sample_weight: ArrayLike | None,
    rng: np.random.Generator | None,
    batch: int | None,
) -> tuple[object, NDArray | None]:
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if batch is not None and batch < 1:
        raise ValueError("batch must be a positive integer or None.")
    weights = validate_and_normalise_weights(
        None if sample_weight is None else np.asarray(sample_weight, dtype=float),
        len(actual),
    )

    def statistic(labels: NDArray[np.int_], scores: NDArray) -> float:
        if weights is None:
            return float(metric(labels, scores))
        return float(metric(labels, scores, sample_weight=weights))

    result = permutation_test(
        data=(actual, predicted),
        statistic=statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        batch=batch,
        alternative=alternative,
        rng=np.random.default_rng() if rng is None else rng,
    )
    return result, weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    options: ShiftOptions = ShiftOptions(),
) -> ShiftDetails:
    """Return full permutation-test details for a shift test."""
    actual, predicted = build_two_sample_dataset(source, target)
    statistic_name, metric = get_shift_metric(statistic)
    _validate_shift_scores(statistic_name, predicted)
    effective_weight = _resolve_weighting(actual, options.weighting)
    result, _ = _run_permutation_test(
        actual,
        predicted,
        metric,
        n_resamples=options.n_resamples,
        alternative=options.alternative,
        sample_weight=effective_weight,
        rng=options.rng,
        batch=options.batch,
    )
    return ShiftDetails(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        statistic_name=statistic_name,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
    )


def test_adverse_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    direction: Direction,
    options: AdverseShiftOptions = AdverseShiftOptions(),
) -> AdverseShiftDetails:
    """Return full permutation-test details for an adverse-shift test."""
    actual, predicted, validated_direction = _prepare_adverse_inputs(
        source=source,
        target=target,
        direction=direction,
    )
    effective_weight = _resolve_weighting(actual, options.weighting)
    result, weights = _run_permutation_test(
        actual,
        predicted,
        wauc,
        n_resamples=options.n_resamples,
        alternative="greater",
        sample_weight=effective_weight,
        rng=options.rng,
        batch=options.batch,
    )
    posterior: NDArray[np.float64] | None = None
    bayes_factor: float | None = None
    if options.bayesian:
        posterior = np.asarray(
            bayesian_posterior(
                actual,
                predicted,
                wauc,
                n_resamples=options.n_resamples,
                rng=options.rng,
                base_weight=weights,
            ),
            dtype=np.float64,
        )
        bayes_factor = float(
            _bayes_factor(posterior, float(np.mean(result.null_distribution)))
        )
    return AdverseShiftDetails(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
        bayes_factor=bayes_factor,
        posterior=posterior,
    )


__all__ = [
    # detailed result types
    "AdverseShiftDetails",
    "ShiftDetails",
    # options types
    "AdverseShiftOptions",
    "ShiftOptions",
    # weighting strategies
    "ContextualRIWWeighting",
    "NoWeighting",
    "SampleWeighting",
    "WeightingStrategy",
    # functions
    "test_adverse_shift",
    "test_shift",
]
