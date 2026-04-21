from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils.multiclass import type_of_target

from samesame._bayesboot import bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._metrics import get_shift_metric, requires_binary_scores, wauc
from samesame._stats import _bayes_factor
from samesame.importance_weights import ContextWeightingMode, contextual_riw
from samesame._utils import (
    Direction,
    validate_and_normalise_weights,
    validate_direction,
)

ShiftStatistic: TypeAlias = Literal[
    "roc_auc",
    "balanced_accuracy",
    "matthews_corrcoef",
]


@dataclass(frozen=True)
class TestResult:
    """Shared summary fields for public test results."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftResult(TestResult):
    """Summary result for the primary shift test."""

    statistic_name: str


@dataclass(frozen=True)
class AdverseShiftResult(TestResult):
    """Summary result for the primary adverse-shift test."""

    direction: Direction


@dataclass(frozen=True)
class ShiftDetails(ShiftResult):
    """Detailed result for the advanced shift test."""

    null_distribution: NDArray[np.float64]

    def summary(self) -> ShiftResult:
        """Return the primary summary view of this detailed result."""
        return ShiftResult(
            statistic=self.statistic,
            pvalue=self.pvalue,
            statistic_name=self.statistic_name,
        )


@dataclass(frozen=True)
class AdverseShiftDetails(AdverseShiftResult):
    """Detailed result for the advanced adverse-shift test."""

    null_distribution: NDArray[np.float64]
    bayes_factor: float | None = None
    posterior: NDArray[np.float64] | None = None

    def summary(self) -> AdverseShiftResult:
        """Return the primary summary view of this detailed result."""
        return AdverseShiftResult(
            statistic=self.statistic,
            pvalue=self.pvalue,
            direction=self.direction,
        )


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


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
        rng=_resolve_rng(rng),
    )
    return result, weights


def _validate_shift_scores(statistic_name: str, predicted: NDArray) -> None:
    if not requires_binary_scores(statistic_name):
        return
    if type_of_target(predicted, "predicted") != "binary":
        raise ValueError(f"statistic={statistic_name!r} requires binary score vectors.")


def _resolve_context_sample_weight(
    *,
    actual: NDArray[np.int_],
    sample_weight: ArrayLike | None,
    context_membership_probabilities: ArrayLike | None,
    context_mode: ContextWeightingMode | None,
    context_lam: float,
    context_prior_ratio: float | None,
) -> ArrayLike | None:
    """Resolve effective weights from either explicit weights or context-aware RIW."""
    if context_membership_probabilities is None and context_mode is None:
        return sample_weight
    if context_membership_probabilities is None or context_mode is None:
        raise ValueError(
            "context_membership_probabilities and context_mode must be provided "
            "together."
        )
    if sample_weight is not None:
        raise ValueError(
            "sample_weight cannot be combined with context-aware weighting inputs."
        )
    context_probs = np.asarray(context_membership_probabilities, dtype=float)
    return contextual_riw(
        actual,
        context_probs,
        mode=context_mode,
        lam=context_lam,
        prior_ratio=context_prior_ratio,
    )


def _prepare_adverse_inputs(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    direction: str,
) -> tuple[NDArray[np.int_], NDArray, Direction]:
    actual, predicted = build_two_sample_dataset(reference, candidate)
    validated_direction = validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    return actual, predicted, validated_direction


def test_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
) -> ShiftResult:
    """Test whether two score vectors differ distributionally.

    Parameters
    ----------
    reference : ArrayLike
        Baseline score vector.
    candidate : ArrayLike
        New score vector to compare against ``reference``.
    statistic : {'roc_auc', 'balanced_accuracy', 'matthews_corrcoef'}, optional
        Named built-in statistic used inside the permutation test.

    Returns
    -------
    ShiftResult
        Immutable summary result containing the test statistic and p-value.
    """
    return _advanced_test_shift(
        reference=reference,
        candidate=candidate,
        statistic=statistic,
        n_resamples=9999,
        sample_weight=None,
        alternative="two-sided",
        rng=None,
        batch=None,
    ).summary()


def test_adverse_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    direction: Direction,
) -> AdverseShiftResult:
    """Test whether the candidate sample is harmfully shifted.

    Parameters
    ----------
    reference : ArrayLike
        Baseline score vector.
    candidate : ArrayLike
        New score vector to compare against ``reference``.
    direction : {'higher-is-worse', 'higher-is-better'}
        Semantic direction of larger score values.

    Returns
    -------
    AdverseShiftResult
        Immutable summary result containing the test statistic and p-value.
    """
    return _advanced_test_adverse_shift(
        reference=reference,
        candidate=candidate,
        direction=direction,
        n_resamples=9999,
        sample_weight=None,
        bayesian=False,
        rng=None,
        batch=None,
    ).summary()


def _advanced_test_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    n_resamples: int = 9999,
    sample_weight: ArrayLike | None = None,
    context_membership_probabilities: ArrayLike | None = None,
    context_mode: ContextWeightingMode | None = None,
    context_lam: float = 0.5,
    context_prior_ratio: float | None = None,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    rng: np.random.Generator | None = None,
    batch: int | None = None,
) -> ShiftDetails:
    """Run the detailed shift test used by the advanced API."""
    actual, predicted = build_two_sample_dataset(reference, candidate)
    statistic_name, metric = get_shift_metric(statistic)
    _validate_shift_scores(statistic_name, predicted)
    effective_weight = _resolve_context_sample_weight(
        actual=actual,
        sample_weight=sample_weight,
        context_membership_probabilities=context_membership_probabilities,
        context_mode=context_mode,
        context_lam=context_lam,
        context_prior_ratio=context_prior_ratio,
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


def _advanced_test_adverse_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    direction: Direction,
    n_resamples: int = 9999,
    sample_weight: ArrayLike | None = None,
    context_membership_probabilities: ArrayLike | None = None,
    context_mode: ContextWeightingMode | None = None,
    context_lam: float = 0.5,
    context_prior_ratio: float | None = None,
    bayesian: bool = False,
    rng: np.random.Generator | None = None,
    batch: int | None = None,
) -> AdverseShiftDetails:
    """Run the detailed adverse-shift test used by the advanced API."""
    actual, predicted, validated_direction = _prepare_adverse_inputs(
        reference=reference,
        candidate=candidate,
        direction=direction,
    )
    effective_weight = _resolve_context_sample_weight(
        actual=actual,
        sample_weight=sample_weight,
        context_membership_probabilities=context_membership_probabilities,
        context_mode=context_mode,
        context_lam=context_lam,
        context_prior_ratio=context_prior_ratio,
    )
    resolved_rng = _resolve_rng(rng)
    result, weights = _run_permutation_test(
        actual,
        predicted,
        wauc,
        n_resamples=n_resamples,
        alternative="greater",
        sample_weight=effective_weight,
        rng=resolved_rng,
        batch=batch,
    )
    posterior: NDArray[np.float64] | None = None
    bayes_factor: float | None = None
    if bayesian:
        posterior = np.asarray(
            bayesian_posterior(
                actual,
                predicted,
                wauc,
                n_resamples=n_resamples,
                rng=resolved_rng,
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
    "AdverseShiftDetails",
    "AdverseShiftResult",
    "ShiftDetails",
    "ShiftResult",
    "TestResult",
    "test_adverse_shift",
    "test_shift",
    "_advanced_test_adverse_shift",
    "_advanced_test_shift",
]
