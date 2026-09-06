"""Domain classification and weighting for harm tests — single seam."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from skrub import tabular_pipeline

from samesame.shift import test_harmful_shift
from samesame.weights import ImportanceWeights, domain_weights
from scripts.style import MODE_ORDER

ALPHA = 0.05
DEFAULT_HGB_PARAMS: dict[str, Any] = dict(
    max_iter=1000, learning_rate=0.05, max_depth=6, min_samples_leaf=20
)

DomainProbabilityEstimator = Callable[
    [Any, Any], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def _as_2d(feature: Any) -> NDArray[np.float64]:
    if isinstance(feature, pl.DataFrame | pl.Series):
        arr = feature.to_numpy()
    else:
        arr = np.asarray(feature, dtype=np.float64)
    return arr.astype(np.float64, copy=False).reshape(len(arr), -1)


def clip_domain_probabilities(p: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _cross_val_probs(est: Any, X: Any, y: NDArray[np.int_], n_source: int, *, cv: int = 10) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    prob = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    clipped = clip_domain_probabilities(prob)
    return clipped[:n_source], clipped[n_source:]


def _hgb_pipeline() -> Any:
    return tabular_pipeline(HistGradientBoostingClassifier(random_state=42, **DEFAULT_HGB_PARAMS))


def cross_fitted_domain_probs(source_feature: Any, target_feature: Any, *, make_estimator: Callable[[], Any], cv: int = 10) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s2d, t2d = _as_2d(source_feature), _as_2d(target_feature)
    X = np.vstack([s2d, t2d])
    y = np.concatenate([np.zeros(len(s2d), dtype=int), np.ones(len(t2d), dtype=int)])
    return _cross_val_probs(make_estimator(), X, y, len(s2d), cv=cv)


def estimate_domain_probabilities_hgb(source_feature: Any, target_feature: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(source_feature, pl.DataFrame) and isinstance(target_feature, pl.DataFrame):
        n_s = source_feature.height
        X = pl.concat([source_feature, target_feature], how="vertical")
        y = np.concatenate([np.zeros(n_s, dtype=int), np.ones(len(X) - n_s, dtype=int)])
        return _cross_val_probs(_hgb_pipeline(), X, y, n_s)
    return cross_fitted_domain_probs(source_feature, target_feature, make_estimator=_hgb_pipeline)


def crump_trimming_mask(s_p: NDArray[np.float64], t_p: NDArray[np.float64], *, threshold: float = 0.1) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    return (
        (np.minimum(s_p, 1 - s_p) >= threshold).astype(bool),
        (np.minimum(t_p, 1 - t_p) >= threshold).astype(bool),
    )


def estimate_overlap_weights(s_p: NDArray[np.float64], t_p: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return s_p * (1 - s_p), t_p * (1 - t_p)


def weight_diagnostics(w: ImportanceWeights | None, *, n_source: int, n_target: int) -> dict[str, float]:
    if w is None:
        return {"source_ess": float(n_source), "target_ess": float(n_target), "source_max_weight": 1.0, "target_max_weight": 1.0}
    ess = w.effective_sample_size()
    return {"source_ess": ess.source, "target_ess": ess.target, "source_max_weight": float(np.asarray(w.source).max()), "target_max_weight": float(np.asarray(w.target).max())}


def run_weighted_harm_test(source_score, target_score, *, direction: str, source_domain_prob, target_domain_prob, mode: str, lambda_value: float, n_resamples: int, seed: int, alpha: float = ALPHA) -> dict[str, float | str]:
    if mode == "crump":
        s_keep, t_keep = crump_trimming_mask(source_domain_prob, target_domain_prob)
        result = test_harmful_shift(source_score[s_keep], target_score[t_keep], worse=direction, weights=None, n_resamples=n_resamples, rng=seed)
        ess = {"source_ess": float(s_keep.sum()), "target_ess": float(t_keep.sum()), "source_max_weight": 1.0, "target_max_weight": 1.0}
    else:
        if mode == "unweighted":
            w = None
        elif mode == "overlap":
            s_ow, t_ow = estimate_overlap_weights(source_domain_prob, target_domain_prob)
            w = ImportanceWeights(source=s_ow, target=t_ow)
        else:
            w = domain_weights(source=source_domain_prob, target=target_domain_prob, reweight=mode, shrinkage=lambda_value)
        result = test_harmful_shift(source_score, target_score, worse=direction, weights=w, n_resamples=n_resamples, rng=seed)
        ess = weight_diagnostics(w, n_source=len(source_score), n_target=len(target_score))
    return {"mode": mode, "statistic": float(result.statistic), "pvalue": float(result.pvalue), "reject": float(result.pvalue < alpha), **ess}


def run_harm_test(source_score, target_score, *, source_feature, target_feature, mode: str, lambda_value: float, n_resamples: int, seed: int, estimator: DomainProbabilityEstimator = estimate_domain_probabilities_hgb, direction: str = "higher", alpha: float = ALPHA) -> dict[str, float | str]:
    s_p, t_p = estimator(source_feature, target_feature)
    return run_weighted_harm_test(source_score, target_score, direction=direction, source_domain_prob=s_p, target_domain_prob=t_p, mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=seed, alpha=alpha)


def run_mode_grid(*, n_repeats: int, values: Sequence[Any], draw: Callable[[Any, int], dict[str, NDArray[np.float64]]], extra: Callable[[Any], dict[str, Any]], lambda_value: float, n_resamples: int, draw_seed: int, test_seed: int) -> list[dict[str, Any]]:
    """Run all weighting modes over repeats x grid values with common random numbers."""
    rows: list[dict[str, Any]] = []
    for rep in range(n_repeats):
        for value in values:
            ds = draw(value, draw_seed + rep)
            for mode in MODE_ORDER:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=test_seed + rep)
                rows.append({"repeat": rep, **extra(value), **r})
    return rows
