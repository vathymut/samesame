"""Weighted two-sample permutation testing (label-permutation null).

The ``+1`` smoothing follows Phipson & Smyth (2010): ``(count+1)/(n+1)``
for one-sided and doubling the smaller tail (capped at 1) for two-sided,
so p-values are never exactly zero.

References
----------
Phipson, B., Smyth, G. K. (2010). Permutation P-values should never be
    zero. *Stat. Appl. Genet. Mol. Biol.* 9(1):Article 39.
    https://doi.org/10.2202/1544-6115.1585
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from samesame.weights import ImportanceWeights

Rng = np.random.Generator | np.random.RandomState
Seed = int | Rng | None


def _resolve_rng(rng: Seed) -> Rng:
    """Normalize *rng* to a ``Generator`` or ``RandomState``."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator | np.random.RandomState):
        return rng
    if isinstance(rng, int | np.integer):
        return np.random.default_rng(int(rng))
    raise TypeError(
        "rng must be an int seed, numpy.random.Generator, numpy.random.RandomState, or None."
    )


def _check_finite(arr: NDArray[np.float64], *, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf).")


def _as_scores(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values)
    # Accept (n,1) or (1,n) column vectors like sklearn's column_or_1d
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise ValueError(f"{name} must be a one-dimensional numeric array.")
    arr = arr.astype(np.float64, copy=False)
    _check_finite(arr, name=name)
    return arr


def _as_sample_weight(
    weights: ImportanceWeights, *, n_source: int, n_target: int
) -> NDArray[np.float64]:
    src = weights.source
    tgt = weights.target
    if src.shape[0] != n_source:
        raise ValueError(
            f"weights.source has wrong length: expected {n_source}, got {src.shape[0]}."
        )
    if tgt.shape[0] != n_target:
        raise ValueError(
            f"weights.target has wrong length: expected {n_target}, got {tgt.shape[0]}."
        )
    return np.concatenate((src, tgt))


def _pvalue(
    observed: float,
    null: NDArray[np.float64],
    alternative: Literal["less", "greater", "two-sided"],
) -> float:
    """Conservative permutation p-value with ``+1`` smoothing.

    Implements the Phipson & Smyth (2010) exact ``(count+1)/(n+1)`` form
    (one-sided) and the two-sided doubling of the smaller tail (capped at
    1). See module References for the full citation.
    """
    n = null.size
    # small epsilon to treat near-equal floats as equal (matches scipy's gamma)
    eps = np.finfo(float).eps * 100
    gamma = abs(eps * observed) if np.isfinite(observed) else 0.0

    if alternative == "greater":
        count = np.sum(null >= observed - gamma)
        return float((count + 1) / (n + 1))
    if alternative == "less":
        count = np.sum(null <= observed + gamma)
        return float((count + 1) / (n + 1))
    # two-sided: double the smaller tail (capped at 1)
    count_greater = np.sum(null >= observed - gamma)
    count_less = np.sum(null <= observed + gamma)
    p_greater = (count_greater + 1) / (n + 1)
    p_less = (count_less + 1) / (n + 1)
    return float(min(1.0, 2 * min(p_greater, p_less)))


def _permutation_test(
    source: ArrayLike,
    target: ArrayLike,
    *,
    metric: Callable[
        [NDArray[np.int_], NDArray[np.float64], NDArray[np.float64] | None], float
    ],
    alternative: Literal["less", "greater", "two-sided"],
    n_resamples: int,
    rng: Seed,
    weights: ImportanceWeights | None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Run a label-permutation test keeping scores (and weights) fixed."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    rng = _resolve_rng(rng)

    source_scores = _as_scores(source, name="source")
    target_scores = _as_scores(target, name="target")

    labels = np.concatenate(
        (
            np.zeros(source_scores.size, dtype=int),
            np.ones(target_scores.size, dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    sample_weight = None
    if weights is not None:
        sample_weight = _as_sample_weight(
            weights, n_source=source_scores.size, n_target=target_scores.size
        )

    observed = float(metric(labels, scores, sample_weight))

    null_distribution = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        # RandomState.permutation vs Generator.permutation have slightly
        # different signatures but both work with array.
        perm_labels = rng.permutation(labels)
        null_distribution[i] = float(metric(perm_labels, scores, sample_weight))

    pvalue = _pvalue(observed, null_distribution, alternative)
    return observed, pvalue, null_distribution
