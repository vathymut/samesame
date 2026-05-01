from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import permutation_test
from sklearn.utils.multiclass import type_of_target

from samesame._bayesboot import bayesian_posterior
from samesame._data import build_two_sample_dataset
from samesame._metrics import get_shift_metric, requires_binary_scores, wauc
from samesame._stats import _bayes_factor
from samesame._utils import (
    Direction,
    validate_and_normalise_weights,
    validate_direction,
)
from samesame.importance_weights import ContextWeightingMode, contextual_riw

ShiftStatistic: TypeAlias = Literal[
    "roc_auc",
    "balanced_accuracy",
    "matthews_corrcoef",
]


# ---------------------------------------------------------------------------
# Weighting strategies — tagged union
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoWeighting:
    """No sample weighting; all observations are treated equally."""


@dataclass(frozen=True)
class SampleWeighting:
    """Explicit per-sample weights.

    Parameters
    ----------
    values : ArrayLike
        Weight for each observation in the pooled (source + target) dataset.
    """

    values: ArrayLike


@dataclass(frozen=True)
class ContextualRIWWeighting:
    """Context-aware Relative Importance Weighting (RIW).

    Parameters
    ----------
    probabilities : ArrayLike
        Membership probabilities P(target | x) in (0, 1) for the pooled dataset.
    mode : ContextWeightingMode
        Weighting strategy — one of ``'source-reweighting'``,
        ``'target-reweighting'``, or
        ``'double-weighting-covariate-shift-adaptation'``.
    lam : float, optional
        RIW blending parameter in [0, 1], by default ``0.5``.
    prior_ratio : float or None, optional
        Ratio n_source / n_target for prior correction. Inferred when ``None``.
    """

    probabilities: ArrayLike
    mode: ContextWeightingMode
    lam: float = 0.5
    prior_ratio: float | None = None


WeightingStrategy = Union[NoWeighting, SampleWeighting, ContextualRIWWeighting]


# ---------------------------------------------------------------------------
# Options dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResamplingOptions:
    """Shared resampling controls.

    Parameters
    ----------
    n_resamples : int, optional
        Number of permutation resamples, by default ``9999``.
    batch : int or None, optional
        Number of resamples to process per batch. ``None`` uses a single batch.
    rng : numpy.random.Generator or None, optional
        Random number generator for reproducibility. ``None`` creates a fresh one.
    """

    n_resamples: int = 9999
    batch: int | None = None
    rng: np.random.Generator | None = None


@dataclass(frozen=True)
class ShiftOptions(ResamplingOptions):
    """Options for :func:`advanced.test_shift`.

    Parameters
    ----------
    weighting : WeightingStrategy, optional
        Weighting strategy. Defaults to :class:`NoWeighting`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Alternative hypothesis for the permutation test, by default ``'two-sided'``.
    """

    weighting: WeightingStrategy = field(default_factory=NoWeighting)
    alternative: Literal["less", "greater", "two-sided"] = "two-sided"


@dataclass(frozen=True)
class AdverseShiftOptions(ResamplingOptions):
    """Options for :func:`advanced.test_adverse_shift`.

    Parameters
    ----------
    weighting : WeightingStrategy, optional
        Weighting strategy. Defaults to :class:`NoWeighting`.
    bayesian : bool, optional
        When ``True``, also compute the Bayes factor and posterior draws,
        by default ``False``.
    """

    weighting: WeightingStrategy = field(default_factory=NoWeighting)
    bayesian: bool = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_weighting(
    actual: NDArray[np.int_], weighting: WeightingStrategy
) -> ArrayLike | None:
    if isinstance(weighting, NoWeighting):
        return None
    if isinstance(weighting, SampleWeighting):
        return validate_and_normalise_weights(
            np.asarray(weighting.values, dtype=float), len(actual)
        )
    if isinstance(weighting, ContextualRIWWeighting):
        return contextual_riw(
            actual,
            np.asarray(weighting.probabilities, dtype=float),
            mode=weighting.mode,
            lam=weighting.lam,
            prior_ratio=weighting.prior_ratio,
        )
    raise TypeError(f"Unsupported weighting strategy: {type(weighting)}")


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


# ---------------------------------------------------------------------------
# Primary (simple) API
# ---------------------------------------------------------------------------


def test_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    options: ShiftOptions = ShiftOptions(),
) -> ShiftResult:
    """Test whether two outlier score distributions differ.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
    statistic : {'roc_auc', 'balanced_accuracy', 'matthews_corrcoef'}, optional
        Named built-in statistic used inside the permutation test.
    options : ShiftOptions, optional
        Resampling, weighting, and alternative-hypothesis controls.

    Returns
    -------
    ShiftResult
        Immutable summary containing the test statistic and p-value.
    """
    return _advanced_test_shift(
        source=source,
        target=target,
        statistic=statistic,
        options=options,
    ).summary()


def test_adverse_shift(
    *,
    source: ArrayLike,
    target: ArrayLike,
    direction: Direction,
    options: AdverseShiftOptions = AdverseShiftOptions(),
) -> AdverseShiftResult:
    """Test whether the target sample is harmfully shifted.

    Parameters
    ----------
    source : ArrayLike
        Baseline outlier scores.
    target : ArrayLike
        New outlier scores to compare against ``source``.
    direction : {'higher-is-worse', 'higher-is-better'}
        Semantic direction of larger score values.
    options : AdverseShiftOptions, optional
        Resampling, weighting, and Bayesian controls.

    Returns
    -------
    AdverseShiftResult
        Immutable summary containing the test statistic and p-value.
    """
    return _advanced_test_adverse_shift(
        source=source,
        target=target,
        direction=direction,
        options=options,
    ).summary()


# ---------------------------------------------------------------------------
# Advanced (detailed) API  — used directly by samesame.advanced
# ---------------------------------------------------------------------------


def _advanced_test_shift(
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


def _advanced_test_adverse_shift(
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
    # result types
    "AdverseShiftDetails",
    "AdverseShiftResult",
    "ShiftDetails",
    "ShiftResult",
    "TestResult",
    # options types
    "AdverseShiftOptions",
    "ShiftOptions",
    # weighting strategies
    "AdverseShiftOptions",
    "ContextualRIWWeighting",
    "NoWeighting",
    "SampleWeighting",
    "WeightingStrategy",
    # public functions
    "test_adverse_shift",
    "test_shift",
    # advanced internals (re-exported via samesame.advanced)
    "_advanced_test_adverse_shift",
    "_advanced_test_shift",
]
