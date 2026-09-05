"""Weight-based harm-test dispatch for manuscript experiments.

This module owns the dispatch logic that connects a mode name (unweighted,
source, target, both, crump, overlap) to the correct weighting and harm-test
path. Domain-probability estimation lives in ``_domain_clf.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from samesame.shift import HarmfulShiftResult, test_harmful_shift
from samesame.weights import ImportanceWeights, domain_weights
from scripts._domain_clf import DomainProbabilityEstimator


def crump_trimming_mask(
    source_domain_prob: NDArray[np.float64],
    target_domain_prob: NDArray[np.float64],
    *,
    threshold: float = 0.1,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    source_mask = np.minimum(source_domain_prob, 1.0 - source_domain_prob) >= threshold
    target_mask = np.minimum(target_domain_prob, 1.0 - target_domain_prob) >= threshold
    return source_mask.astype(bool), target_mask.astype(bool)


def weight_diagnostics(
    weights: ImportanceWeights | None,
    *,
    n_source: int,
    n_target: int,
) -> dict[str, float]:
    if weights is None:
        return {
            "source_ess": float(n_source),
            "target_ess": float(n_target),
            "source_max_weight": 1.0,
            "target_max_weight": 1.0,
        }
    # Use package method for ESS calculation
    ess = weights.effective_sample_size()
    source_weight = np.asarray(weights.source, dtype=np.float64)
    target_weight = np.asarray(weights.target, dtype=np.float64)
    return {
        "source_ess": ess.source,
        "target_ess": ess.target,
        "source_max_weight": float(source_weight.max()),
        "target_max_weight": float(target_weight.max()),
    }


def run_crump_harm_test(
    source_score: NDArray[np.float64],
    target_score: NDArray[np.float64],
    *,
    direction: str,
    source_domain_prob: NDArray[np.float64],
    target_domain_prob: NDArray[np.float64],
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, float | str]:
    source_mask, target_mask = crump_trimming_mask(
        source_domain_prob, target_domain_prob
    )
    trimmed_source = source_score[source_mask]
    trimmed_target = target_score[target_mask]
    result: HarmfulShiftResult = test_harmful_shift(
        trimmed_source,
        trimmed_target,
        worse=direction,
        weights=None,
        n_resamples=n_resamples,
        rng=seed,
    )
    return {
        "mode": "crump",
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "reject": float(result.pvalue < alpha),
        "source_ess": float(source_mask.sum()),
        "target_ess": float(target_mask.sum()),
        "source_max_weight": 1.0,
        "target_max_weight": 1.0,
    }


def build_weights_from_domain_probabilities(
    *,
    source_domain_prob: NDArray[np.float64],
    target_domain_prob: NDArray[np.float64],
    mode: str,
    lambda_value: float,
) -> ImportanceWeights | None:
    if mode == "unweighted":
        return None
    return domain_weights(
        source=source_domain_prob,
        target=target_domain_prob,
        reweight=mode,
        shrinkage=lambda_value,
    )


def estimate_overlap_weights(
    source_domain_prob: NDArray[np.float64],
    target_domain_prob: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source_weights = source_domain_prob * (1.0 - source_domain_prob)
    target_weights = target_domain_prob * (1.0 - target_domain_prob)
    return source_weights, target_weights


def run_weighted_harm_test(
    source_score: NDArray[np.float64],
    target_score: NDArray[np.float64],
    *,
    direction: str,
    source_domain_prob: NDArray[np.float64],
    target_domain_prob: NDArray[np.float64],
    mode: str,
    lambda_value: float,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, float | str]:
    if mode == "crump":
        return run_crump_harm_test(
            source_score,
            target_score,
            direction=direction,
            source_domain_prob=source_domain_prob,
            target_domain_prob=target_domain_prob,
            n_resamples=n_resamples,
            seed=seed,
            alpha=alpha,
        )
    if mode == "overlap":
        source_ow, target_ow = estimate_overlap_weights(
            source_domain_prob, target_domain_prob
        )
        weights = ImportanceWeights(source=source_ow, target=target_ow)
    else:
        weights = build_weights_from_domain_probabilities(
            source_domain_prob=source_domain_prob,
            target_domain_prob=target_domain_prob,
            mode=mode,
            lambda_value=lambda_value,
        )
    result: HarmfulShiftResult = test_harmful_shift(
        source_score,
        target_score,
        worse=direction,
        weights=weights,
        n_resamples=n_resamples,
        rng=seed,
    )
    diagnostics = weight_diagnostics(
        weights,
        n_source=len(source_score),
        n_target=len(target_score),
    )
    return {
        "mode": mode,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "reject": float(result.pvalue < alpha),
        **diagnostics,
    }


def run_harm_test_with_estimator(
    source_score: NDArray[np.float64],
    target_score: NDArray[np.float64],
    *,
    source_feature: Any,
    target_feature: Any,
    estimator: DomainProbabilityEstimator,
    direction: str,
    mode: str,
    lambda_value: float,
    n_resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, float | str]:
    source_domain_prob, target_domain_prob = estimator(source_feature, target_feature)
    return run_weighted_harm_test(
        source_score,
        target_score,
        direction=direction,
        source_domain_prob=source_domain_prob,
        target_domain_prob=target_domain_prob,
        mode=mode,
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
    )
