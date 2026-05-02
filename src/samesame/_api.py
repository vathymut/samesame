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

from samesame._bayesboot import bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._metrics import get_shift_metric, requires_binary_scores, wauc
from samesame._stats import _bayes_factor
from samesame._types import (
    AdverseShiftDetails,
    ShiftDetails,
    ShiftStatistic,
    TestResult,
)
from samesame._utils import (
    Direction,
    validate_and_normalise_weights,
    validate_direction,
)
from samesame.weights import WeightingMode, contextual_weights


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_weights(
    actual: NDArray,
    weights: ArrayLike | None,
    membership_prob: ArrayLike | None,
    mode: WeightingMode,
    alpha_blend: float,
) -> NDArray | None:
    if weights is not None and membership_prob is not None:
        raise ValueError("Provide either weights or membership_prob, not both.")
    if weights is not None:
        return validate_and_normalise_weights(
            np.asarray(weights, dtype=float), len(actual)
        )
    if membership_prob is not None:
        return contextual_weights(
            actual,
            np.asarray(membership_prob, dtype=float),
            mode=mode,
            alpha_blend=alpha_blend,
        )
    return None


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
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    n_resamples: int = 9999,
    batch: int | None = None,
    rng: np.random.Generator | None = None,
    weights: ArrayLike | None = None,
    membership_prob: ArrayLike | None = None,
    mode: WeightingMode = "source",
    alpha_blend: float = 0.5,
) -> ShiftDetails:
    """Test whether two outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
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
    weights : ArrayLike or None, optional
        Explicit per-sample weights for the pooled (source + target)
        dataset. Mutually exclusive with ``membership_prob``.
    membership_prob : ArrayLike or None, optional
        Per-sample probability of belonging to the target group, in (0, 1).
        When supplied, context-aware RIW weights are computed automatically.
        Mutually exclusive with ``weights``.
    mode : {'source', 'target', 'both'}, optional
        Which group to reweight when ``membership_prob`` is given.
        Ignored when ``membership_prob`` is ``None``.
    alpha_blend : float, optional
        Blending parameter controlling numerical stability of RIW weights.
        Only used when ``membership_prob`` is given. Default is ``0.5``.

    Returns
    -------
    ShiftDetails
        Immutable result with ``statistic``, ``pvalue``, ``statistic_name``,
        and ``null_distribution``.
    """
    actual, predicted = build_two_sample_dataset(source, target)
    statistic_name, metric = get_shift_metric(statistic)
    _validate_shift_scores(statistic_name, predicted)
    effective_weight = _resolve_weights(
        actual, weights, membership_prob, mode, alpha_blend
    )
    result, _ = _run_permutation_test(
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
    bayesian: bool = False,
    weights: ArrayLike | None = None,
    membership_prob: ArrayLike | None = None,
    mode: WeightingMode = "source",
    alpha_blend: float = 0.5,
) -> AdverseShiftDetails:
    """Test whether the target sample is harmfully shifted.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
    direction : {'higher-is-worse', 'higher-is-better'}
        Semantic direction of larger score values.
    n_resamples : int, optional
        Number of permutation resamples, by default ``9999``.
    batch : int or None, optional
        Number of resamples to process per batch. ``None`` uses a single
        batch.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a
        fresh one.
    bayesian : bool, optional
        When ``True``, also compute the Bayes factor and posterior draws.
        Default is ``False``.
    weights : ArrayLike or None, optional
        Explicit per-sample weights for the pooled dataset. Mutually
        exclusive with ``membership_prob``.
    membership_prob : ArrayLike or None, optional
        Per-sample probability of belonging to the target group, in (0, 1).
        When supplied, context-aware RIW weights are computed automatically.
        Mutually exclusive with ``weights``.
    mode : {'source', 'target', 'both'}, optional
        Which group to reweight when ``membership_prob`` is given.
    alpha_blend : float, optional
        Blending parameter controlling numerical stability of RIW weights.
        Only used when ``membership_prob`` is given. Default is ``0.5``.

    Returns
    -------
    AdverseShiftDetails
        Immutable result with ``statistic``, ``pvalue``, ``direction``,
        ``null_distribution``, and optionally ``bayes_factor`` and
        ``posterior``.
    """
    actual, predicted = build_two_sample_dataset(source, target)
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    effective_weight = _resolve_weights(
        actual, weights, membership_prob, mode, alpha_blend
    )
    result, resolved_weights = _run_permutation_test(
        actual,
        predicted,
        wauc,
        n_resamples=n_resamples,
        alternative="greater",
        sample_weight=effective_weight,
        rng=rng,
        batch=batch,
    )
    posterior: NDArray[np.float64] | None = None
    bayes_factor_val: float | None = None
    if bayesian:
        posterior = np.asarray(
            bayesian_posterior(
                actual,
                predicted,
                wauc,
                n_resamples=n_resamples,
                rng=rng,
                base_weight=resolved_weights,
            ),
            dtype=np.float64,
        )
        bayes_factor_val = float(
            _bayes_factor(posterior, float(np.mean(result.null_distribution)))
        )
    return AdverseShiftDetails(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
        bayes_factor=bayes_factor_val,
        posterior=posterior,
    )


__all__ = [
    "AdverseShiftDetails",
    "ShiftDetails",
    "TestResult",
    "test_adverse_shift",
    "test_shift",
]
