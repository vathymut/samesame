"""Public shift-detection seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.stats import permutation_test
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target

from samesame.weights import ImportanceWeights

Direction = Literal["higher-is-worse", "higher-is-better"]
RandomState = int | np.random.RandomState | np.random.Generator | None
type ShiftStatistic = Literal["roc_auc", "balanced_accuracy", "matthews_corrcoef"]


@dataclass(frozen=True)
class TestResult:
    """Shared fields for all statistical test results."""

    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftResult(TestResult):
    """Result of generic shift detection."""

    statistic_name: str
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class HarmResult(TestResult):
    """Result of harmful-shift detection."""

    direction: Direction
    null_distribution: NDArray[np.float64]
    posterior: NDArray[np.float64] | None = None
    bayes_factor: float | None = None


@dataclass(frozen=True)
class _TwoSampleDataset:
    labels: NDArray[np.int_]
    scores: NDArray
    n_source: int
    n_target: int


class _ECDFDiscrete:
    def __init__(self, x: NDArray, freq_weights: NDArray | None = None, side: str = "right"):
        if side.lower() not in ["right", "left"]:
            raise ValueError(f"side can take the values 'right' or 'left', got {side}")
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")
        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
            if freq_weights.ndim != 1:
                raise ValueError("freq_weights must be one-dimensional.")
            if len(freq_weights) != len(x):
                raise ValueError("freq_weights must have the same length as x.")
            if np.any(freq_weights < 0):
                raise ValueError("freq_weights must be non-negative.")
            order = np.argsort(x)
            x_sorted = x[order]
            w_sorted = freq_weights[order]
            x_unique, first = np.unique(x_sorted, return_index=True)
            w_sum = np.add.reduceat(w_sorted, first)
            y = np.cumsum(w_sum) / np.sum(w_sum)
            x = x_unique
        else:
            x = np.sort(x)
            y = np.linspace(1.0 / len(x), 1.0, len(x))
        self.side = side
        self.x = np.r_[-np.inf, x]
        self.y = np.r_[0.0, y]

    def __call__(self, time: NDArray) -> NDArray[np.float64]:
        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


_SHIFT_STATISTICS: dict[str, Callable[..., float]] = {
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "matthews_corrcoef": matthews_corrcoef,
}

_BINARY_ONLY_STATISTICS = frozenset({"balanced_accuracy", "matthews_corrcoef"})


def _as_numeric_vector(values: ArrayLike, *, name: str) -> NDArray:
    vector = column_or_1d(values)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (
        np.issubdtype(vector.dtype, np.number) or np.issubdtype(vector.dtype, np.bool_)
    ):
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    return np.asarray(vector)


def _build_two_sample_dataset(source: ArrayLike, target: ArrayLike) -> _TwoSampleDataset:
    source_scores = _as_numeric_vector(source, name="source")
    target_scores = _as_numeric_vector(target, name="target")
    labels = np.concatenate(
        (
            np.zeros(source_scores.shape[0], dtype=int),
            np.ones(target_scores.shape[0], dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    return _TwoSampleDataset(
        labels=labels,
        scores=scores,
        n_source=int(source_scores.shape[0]),
        n_target=int(target_scores.shape[0]),
    )


def _validate_direction(direction: str) -> Direction:
    if direction not in ("higher-is-worse", "higher-is-better"):
        raise ValueError(
            "direction must be one of 'higher-is-worse' or 'higher-is-better'."
        )
    return direction


def _validate_and_normalise_weights(sample_weight: NDArray | None, n: int) -> NDArray | None:
    if sample_weight is None:
        return None
    weight = np.asarray(sample_weight, dtype=float)
    if len(weight) != n:
        raise ValueError(f"sample_weight has wrong length: expected {n}, got {len(weight)}.")
    if not np.all(np.isfinite(weight)):
        raise ValueError("sample_weight must contain only finite values (no NaN or inf).")
    if np.any(weight < 0):
        raise ValueError("sample_weight must not contain negative values.")
    total = weight.sum()
    if total == 0:
        raise ValueError("sample_weight must not be all zero.")
    return weight / total * n


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


def _get_shift_statistic(name: str) -> tuple[str, Callable[..., float]]:
    statistic = _SHIFT_STATISTICS.get(name)
    if statistic is None:
        allowed = ", ".join(sorted(_SHIFT_STATISTICS))
        raise ValueError(f"statistic must be one of {allowed}; got {name!r}.")
    return name, statistic


def _requires_binary_scores(name: str) -> bool:
    return name in _BINARY_ONLY_STATISTICS


def _run_permutation_test(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable[..., float],
    *,
    n_resamples: int,
    alternative: Literal["less", "greater", "two-sided"],
    sample_weight: ArrayLike | None,
    rng: np.random.Generator | np.random.RandomState,
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
        rng=rng,
    )


def _wauc(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    sample_weight: NDArray | None = None,
) -> float:
    fpr, tpr, thresholds = roc_curve(
        actual,
        predicted,
        pos_label=None,
        sample_weight=sample_weight,
    )
    negative_mask = actual == 0
    negative_scores = predicted[negative_mask]
    if sample_weight is None:
        ewcdf = _ECDFDiscrete(negative_scores)
    else:
        negative_weights = sample_weight[negative_mask]
        ewcdf = _ECDFDiscrete(negative_scores, freq_weights=negative_weights)
    weights = np.power(ewcdf(thresholds), 2)
    return float(trapezoid(y=tpr * weights, x=fpr))


def _draw_uniform_dirichlet(
    size: int,
    rng: np.random.Generator | np.random.RandomState,
) -> NDArray:
    return rng.dirichlet(alpha=np.ones(size))


def _bayesian_bootstrap(
    statistic: Callable[[NDArray], float],
    n_obs: int,
    *,
    n_resamples: int,
    rng: np.random.Generator | np.random.RandomState,
) -> NDArray[np.float64]:
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        draws[idx] = statistic(_draw_uniform_dirichlet(n_obs, rng))
    return draws


def _bayesian_posterior(
    actual: NDArray[np.int_],
    predicted: NDArray,
    metric: Callable[..., float],
    *,
    n_resamples: int,
    rng: np.random.Generator | np.random.RandomState,
    base_weight: NDArray | None = None,
) -> NDArray[np.float64]:
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")

    def statistic(sample_weight: NDArray) -> float:
        if base_weight is not None:
            sample_weight = sample_weight * base_weight
        return float(metric(actual, predicted, sample_weight=sample_weight))

    return _bayesian_bootstrap(
        statistic,
        len(actual),
        n_resamples=n_resamples,
        rng=rng,
    )


def _bayes_factor(posterior: NDArray[np.float64], threshold: float) -> float:
    if posterior.ndim != 1:
        raise ValueError("posterior must be one-dimensional.")
    pvalue = float(np.mean(posterior > threshold))
    if pvalue == 1.0:
        return np.inf
    return float(pvalue / (1.0 - pvalue))


def _combine_importance_weights(
    weights: ImportanceWeights | None,
    *,
    n_source: int,
    n_target: int,
) -> NDArray | None:
    if weights is None:
        return None
    source_w = _validate_and_normalise_weights(
        np.asarray(weights.source, dtype=float), n_source
    )
    target_w = _validate_and_normalise_weights(
        np.asarray(weights.target, dtype=float), n_target
    )
    return np.concatenate([source_w, target_w])


def _validate_shift_scores(statistic_name: str, predicted: NDArray) -> None:
    if not _requires_binary_scores(statistic_name):
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
    dataset = _build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    validated_direction = _validate_direction(direction)
    if validated_direction == "higher-is-better":
        predicted = -predicted
    combined_weights = _combine_importance_weights(
        weights,
        n_source=dataset.n_source,
        n_target=dataset.n_target,
    )
    return actual, predicted, validated_direction, combined_weights


def _resolve_posterior_threshold(
    *,
    include_posterior: bool,
    threshold: float | None,
) -> float | None:
    if not include_posterior:
        if threshold is not None:
            raise ValueError("threshold is only valid when include_posterior=True.")
        return None
    if threshold is None:
        return 1 / 12
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    return threshold_value


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
    dataset = _build_two_sample_dataset(source, target)
    actual, predicted = dataset.labels, dataset.scores
    statistic_name, metric = _get_shift_statistic(statistic)
    _validate_shift_scores(statistic_name, predicted)
    combined_weights = _combine_importance_weights(
        weights,
        n_source=dataset.n_source,
        n_target=dataset.n_target,
    )
    result = _run_permutation_test(
        actual,
        predicted,
        metric,
        n_resamples=n_resamples,
        alternative=alternative,
        sample_weight=combined_weights,
        rng=_resolve_random_state(random_state),
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
    include_posterior: bool = False,
    threshold: float | None = None,
) -> HarmResult:
    """Detect whether Target is harmfully shifted relative to Source."""
    actual, predicted, validated_direction, combined_weights = _prepare_harm_detection(
        source, target, direction, weights
    )
    posterior_threshold = _resolve_posterior_threshold(
        include_posterior=include_posterior,
        threshold=threshold,
    )
    result = _run_permutation_test(
        actual,
        predicted,
        _wauc,
        n_resamples=n_resamples,
        alternative="greater",
        sample_weight=combined_weights,
        rng=_resolve_random_state(random_state),
        batch=batch,
    )
    posterior = None
    bayes_factor = None
    if include_posterior:
        posterior = np.asarray(
            _bayesian_posterior(
                actual,
                predicted,
                _wauc,
                n_resamples=n_resamples,
                rng=_resolve_random_state(random_state),
                base_weight=combined_weights,
            ),
            dtype=np.float64,
        )
        bayes_factor = float(_bayes_factor(posterior, posterior_threshold))
    return HarmResult(
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        direction=validated_direction,
        null_distribution=np.asarray(result.null_distribution, dtype=np.float64),
        posterior=posterior,
        bayes_factor=bayes_factor,
    )


__all__ = [
    "Direction",
    "HarmResult",
    "ShiftResult",
    "ShiftStatistic",
    "TestResult",
    "detect_harm",
    "detect_shift",
]
