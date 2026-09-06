"""Domain classification and weighting for harm tests — single seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from skrub import tabular_pipeline

from samesame.shift import HarmfulShiftResult, test_harmful_shift
from samesame.weights import ImportanceWeights, domain_weights
from scripts.style import MODE_ORDER

ALPHA = 0.05
MODES: tuple[str, ...] = MODE_ORDER
DEFAULT_HGB_PARAMS: dict[str, Any] = dict(
    max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20
)
DEFAULT_DOMAIN_CV = 10

DomainProbabilityEstimator = Callable[
    [Any, Any], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def _as_2d(feature: Any) -> NDArray[np.float64]:
    arr = np.asarray(feature, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def clip_domain_probabilities(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def estimate_domain_probabilities_hgb(
    source_feature: Any, target_feature: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s2d, t2d = _as_2d(source_feature), _as_2d(target_feature)
    X = np.vstack([s2d, t2d])
    y = np.concatenate([np.zeros(len(s2d), dtype=int), np.ones(len(t2d), dtype=int)])
    folds = min(DEFAULT_DOMAIN_CV, int(np.ceil(len(X) / 2)))
    est = HistGradientBoostingClassifier(random_state=42, **DEFAULT_HGB_PARAMS)
    prob = cross_val_predict(tabular_pipeline(est), X, y, cv=folds, method="predict_proba")[:, 1]
    clipped = clip_domain_probabilities(prob)
    return clipped[: len(s2d)], clipped[len(s2d) :]


def crump_trimming_mask(
    s_p: NDArray[np.float64], t_p: NDArray[np.float64], *, threshold: float = 0.1
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    return (
        (np.minimum(s_p, 1 - s_p) >= threshold).astype(bool),
        (np.minimum(t_p, 1 - t_p) >= threshold).astype(bool),
    )


def estimate_overlap_weights(
    s_p: NDArray[np.float64], t_p: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return s_p * (1 - s_p), t_p * (1 - t_p)


def weight_diagnostics(
    w: ImportanceWeights | None, *, n_source: int, n_target: int
) -> dict[str, float]:
    if w is None:
        return {"source_ess": float(n_source), "target_ess": float(n_target), "source_max_weight": 1.0, "target_max_weight": 1.0}
    ess = w.effective_sample_size()
    return {
        "source_ess": ess.source,
        "target_ess": ess.target,
        "source_max_weight": float(np.asarray(w.source).max()),
        "target_max_weight": float(np.asarray(w.target).max()),
    }


def _build_weights(s_p, t_p, mode: str, lam: float) -> ImportanceWeights | None:
    if mode == "unweighted":
        return None
    return domain_weights(source=s_p, target=t_p, reweight=mode, shrinkage=lam)


def run_weighted_harm_test(
    source_score, target_score, *, direction: str, source_domain_prob, target_domain_prob,
    mode: str, lambda_value: float, n_resamples: int, seed: int, alpha: float = ALPHA,
) -> dict[str, float | str]:
    if mode == "crump":
        s_mask, t_mask = crump_trimming_mask(source_domain_prob, target_domain_prob)
        result: HarmfulShiftResult = test_harmful_shift(
            source_score[s_mask], target_score[t_mask], worse=direction, weights=None, n_resamples=n_resamples, rng=seed
        )
        return {"mode": "crump", "statistic": float(result.statistic), "pvalue": float(result.pvalue), "reject": float(result.pvalue < alpha), "source_ess": float(s_mask.sum()), "target_ess": float(t_mask.sum()), "source_max_weight": 1.0, "target_max_weight": 1.0}
    if mode == "overlap":
        s_ow, t_ow = estimate_overlap_weights(source_domain_prob, target_domain_prob)
        w = ImportanceWeights(source=s_ow, target=t_ow)
    else:
        w = _build_weights(source_domain_prob, target_domain_prob, mode, lambda_value)
    result: HarmfulShiftResult = test_harmful_shift(source_score, target_score, worse=direction, weights=w, n_resamples=n_resamples, rng=seed)
    diag = weight_diagnostics(w, n_source=len(source_score), n_target=len(target_score))
    return {"mode": mode, "statistic": float(result.statistic), "pvalue": float(result.pvalue), "reject": float(result.pvalue < alpha), **diag}


def run_harm_test_with_estimator(
    source_score, target_score, *, source_feature, target_feature,
    estimator: DomainProbabilityEstimator, direction: str, mode: str,
    lambda_value: float, n_resamples: int, seed: int, alpha: float = ALPHA,
) -> dict[str, float | str]:
    s_p, t_p = estimator(source_feature, target_feature)
    return run_weighted_harm_test(source_score, target_score, direction=direction, source_domain_prob=s_p, target_domain_prob=t_p, mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=seed, alpha=alpha)


def run_harm_test(source_score, target_score, *, source_feature, target_feature, mode: str, lambda_value: float, n_resamples: int, seed: int) -> dict:
    return run_harm_test_with_estimator(source_score, target_score, source_feature=source_feature, target_feature=target_feature, estimator=estimate_domain_probabilities_hgb, direction="higher", mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=seed, alpha=ALPHA)
