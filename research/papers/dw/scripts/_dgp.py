"""Data-generating processes for manuscript experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def draw_overlap_dataset(
    *,
    n_source: int,
    n_target: int,
    source_private_fraction: float,
    target_private_fraction: float,
    target_shared_shift: float,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    source_private = rng.random(n_source) < source_private_fraction
    target_private = rng.random(n_target) < target_private_fraction

    source_feature = rng.normal(loc=0.0, scale=1.0, size=n_source)
    target_feature = rng.normal(loc=0.0, scale=1.0, size=n_target)
    source_feature[source_private] = rng.normal(
        loc=-3.0, scale=0.45, size=source_private.sum()
    )
    target_feature[target_private] = rng.normal(
        loc=3.0, scale=0.45, size=target_private.sum()
    )

    source_score = source_feature + rng.normal(loc=0.0, scale=0.8, size=n_source)
    target_score = target_feature + rng.normal(loc=0.0, scale=0.8, size=n_target)
    target_score[~target_private] += target_shared_shift

    return {
        "source_feature": source_feature,
        "target_feature": target_feature,
        "source_score": source_score,
        "target_score": target_score,
        "source_private_fraction": np.array(
            [source_private_fraction], dtype=np.float64
        ),
        "target_private_fraction": np.array(
            [target_private_fraction], dtype=np.float64
        ),
    }


def draw_second_dgp(
    *,
    n_source: int,
    n_target: int,
    overlap_severity: float,
    effect_size: float,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    source_feature = rng.normal(loc=0.0, scale=1.0, size=(n_source, 2))

    n_private = int(n_target * overlap_severity)
    n_shared = n_target - n_private
    target_feature = np.zeros((n_target, 2))
    target_feature[:n_shared] = rng.normal(loc=0.0, scale=1.0, size=(n_shared, 2))
    target_feature[n_shared:] = rng.normal(loc=3.0, scale=0.5, size=(n_private, 2))

    beta = np.array([0.5, -0.3])
    source_score = source_feature @ beta + rng.normal(0.0, 0.5, size=n_source)
    target_score_shared = target_feature[:n_shared] @ beta + effect_size
    target_score_private = target_feature[n_shared:] @ beta + rng.normal(
        0.0, 0.5, size=n_private
    )
    target_score = np.concatenate([target_score_shared, target_score_private])

    return {
        "source_feature": source_feature[:, 0],
        "target_feature": target_feature[:, 0],
        "source_score": source_score,
        "target_score": target_score,
    }
