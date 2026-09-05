"""Domain-probability estimators for manuscript experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from skrub import tabular_pipeline

DEFAULT_HGB_PARAMS: dict[str, Any] = dict(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
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


def clip_domain_probabilities(probability: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )


def _domain_labels(n_source: int, n_target: int) -> NDArray[np.int_]:
    return np.concatenate(
        [np.zeros(n_source, dtype=int), np.ones(n_target, dtype=int)]
    )


def estimate_domain_probabilities_hgb(
    source_feature: Any,
    target_feature: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source_2d = _as_2d(source_feature)
    target_2d = _as_2d(target_feature)
    feature = np.vstack([source_2d, target_2d])
    group = _domain_labels(len(source_2d), len(target_2d))
    folds = min(DEFAULT_DOMAIN_CV, int(np.ceil(len(feature) / 2)))
    estimator = HistGradientBoostingClassifier(
        random_state=42,
        **DEFAULT_HGB_PARAMS,
    )
    pipeline = tabular_pipeline(estimator)
    probability = cross_val_predict(
        pipeline,
        feature,
        group,
        cv=folds,
        method="predict_proba",
    )[:, 1]
    clipped = clip_domain_probabilities(probability)
    n_source = len(source_2d)
    return clipped[:n_source], clipped[n_source:]
