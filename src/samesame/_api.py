"""Core implementation for hypothesis tests over outlier scores.

The primary API exposes two functions:

- ``test_shift`` for generic score-distribution differences
- ``test_adverse_shift`` for harmful score shifts with explicit direction

Both functions return their full result (including the null distribution)
directly. No separate ``advanced`` sub-module is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils.multiclass import type_of_target

from samesame._bayesboot import _bayes_factor, bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._metrics import get_shift_metric, requires_binary_scores, wauc
from samesame._types import (
    AdverseShiftDetails,
    BayesianEvidence,
    ShiftDetails,
    ShiftStatistic,
    TestResult,
)
from samesame._utils import (
    Direction,
    validate_and_normalise_weights,
    validate_direction,
)
from samesame.weights import ContextualWeights

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_weights(
    weights: ContextualWeights | None,
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
) -> object:
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    if batch is not None and batch < 1:
        raise ValueError("batch must be a positive integer or None.")
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def statistic(labels: NDArray[np.int_], scores: NDArray) -> float:
        if weights is None:
            return float(metric(labels, scores))
        return float(metric(labels, scores, sample_weight=weights))

    return permutation_test(
        data=(actual, predicted),
        statistic=statistic,
        permutation_type="pairings",
        n_resamples=n_resamples,
        batch=batch,
        alternative=alternative,
        rng=np.random.default_rng() if rng is None else rng,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_resamples: int = 9999,
    batch: int | None = None,
    rng: np.random.Generator | None = None,
    weights: ContextualWeights | None = None,
) -> ShiftDetails:
    """Test whether the source and target outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores, typically from training or reference data.
    target : ArrayLike
        New outlier scores to compare against ``source``, typically from
        production or deployment data.
    statistic : {'roc_auc', 'balanced_accuracy', 'matthews_corrcoef'}, optional
        Named built-in statistic used inside the permutation test.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis for the permutation test, by default
        ``'two-sided'``.
    n_resamples : int, optional
        Number of permutation resamples, by default ``9999``.
    batch : int or None, optional
        Number of resamples to process per batch. ``None`` uses a single
        batch.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a
        fresh one.
    weights : ContextualWeights or None, optional
        Importance weights to correct for covariate shift and related concerns
        between source and target. Build from domain probabilities using
        :func:`~samesame.weights.contextual_weights`, or construct
        ``ContextualWeights(source=..., target=...)`` directly.
        Pass ``None`` (default) to run an unweighted test.

    Returns
    -------
    ShiftDetails
        Immutable result with ``statistic``, ``pvalue``, ``statistic_name``,
        and ``null_distribution``.
    """
    dataset = build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    statistic_name, metric = get_shift_metric(statistic)
    _validate_shift_scores(statistic_name, predicted)
    effective_weight = _resolve_weights(weights, dataset.n_source, dataset.n_target)
    result = _run_permutation_test(
        actual,
        predicted,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        sample_weight=effective_weight,
        rng=rng,
        batch=batch,
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
    n_resamples: int = 9999,
    batch: int | None = None,
    rng: np.random.Generator | None = None,
    weights: ContextualWeights | None = None,
) -> AdverseShiftDetails:
    """Test whether the target sample is harmfully shifted.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores, typically from training or reference data.
    target : ArrayLike
        New outlier scores to compare against ``source``, typically from
        production or deployment data.
    direction : {'higher-is-worse', 'higher-is-better'}
        Whether higher outlier scores indicate worse outcomes
        (``'higher-is-worse'``) or better outcomes (``'higher-is-better'``).
        Required to determine the direction of adverse shift.
    n_resamples : int, optional
        Number of permutation resamples, by default ``9999``.
    batch : int or None, optional
        Number of resamples to process per batch. ``None`` uses a single
        batch.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a
        fresh one.
    weights : ContextualWeights or None, optional
        Importance weights to correct for covariate shift and related concerns
        between source and target. Build from domain probabilities using
        :func:`~samesame.weights.contextual_weights`, or construct
        ``ContextualWeights(source=..., target=...)`` directly.
        Pass ``None`` (default) to run an unweighted test.

    Returns
    -------
    AdverseShiftDetails
        Immutable result with ``statistic``, ``pvalue``, ``direction``,
        and ``null_distribution``.

    See Also
    --------
    adverse_shift_posterior : Compute Bayesian evidence on top of this result.
    """
    dataset = build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    effective_weight = _resolve_weights(weights, dataset.n_source, dataset.n_target)
    result = _run_permutation_test(
        actual,
        predicted,
        wauc,
        n_resamples=n_resamples,
        alternative="greater",
        sample_weight=effective_weight,
        rng=rng,
        batch=batch,
    )
    return AdverseShiftDetails(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
    )


def adverse_shift_posterior(
    *,
    source: ArrayLike,
    target: ArrayLike,
    direction: Direction,
    n_resamples: int = 9999,
    rng: np.random.Generator | None = None,
    weights: ContextualWeights | None = None,
    threshold: float = 1 / 12,
) -> BayesianEvidence:
    """Compute Bayesian evidence for adverse shift using a bootstrap posterior.

    Provides a Bayesian evidence layer on top of the adverse-shift test:
    runs a Bayesian bootstrap over the WAUC metric and returns posterior
    draws together with a Bayes factor against a reference threshold.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores, typically from training or reference data.
    target : ArrayLike
        New outlier scores to compare against ``source``, typically from
        production or deployment data.
    direction : {'higher-is-worse', 'higher-is-better'}
        Whether higher outlier scores indicate worse outcomes
        (``'higher-is-worse'``) or better outcomes (``'higher-is-better'``).
        Required to determine the direction of adverse shift.
    n_resamples : int, optional
        Number of Bayesian bootstrap resamples, by default ``9999``.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a
        fresh one.
    weights : ContextualWeights or None, optional
        Importance weights to correct for covariate shift and related concerns
        between source and target. Build from domain probabilities using
        :func:`~samesame.weights.contextual_weights`, or construct
        ``ContextualWeights(source=..., target=...)`` directly.
        Pass ``None`` (default) to run an unweighted test.
    threshold : float, optional
        WAUC value used as the null reference for the Bayes factor.
        Defaults to ``1/12``, the asymptotic expected WAUC under the null
        hypothesis that source and target are from the same distribution.

    Returns
    -------
    BayesianEvidence
        Immutable result with ``posterior`` draws and ``bayes_factor``.

    See Also
    --------
    test_adverse_shift : Run the permutation test for adverse shift.
    """
    dataset = build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    effective_weight = _resolve_weights(weights, dataset.n_source, dataset.n_target)
    posterior = np.asarray(
        bayesian_posterior(
            actual,
            predicted,
            wauc,
            n_resamples=n_resamples,
            rng=rng,
            base_weight=effective_weight,
        ),
        dtype=np.float64,
    )
    bayes_factor_val = float(_bayes_factor(posterior, threshold))
    return BayesianEvidence(
        posterior=posterior,
        bayes_factor=bayes_factor_val,
    )


__all__ = [
    "AdverseShiftDetails",
    "BayesianEvidence",
    "ShiftDetails",
    "TestResult",
    "adverse_shift_posterior",
    "test_adverse_shift",
    "test_shift",
]
