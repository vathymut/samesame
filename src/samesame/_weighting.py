from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from samesame._utils import validate_and_normalise_weights
from samesame.importance_weights import ContextWeightingMode, contextual_riw


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
# Resolver
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


__all__ = [
    "ContextualRIWWeighting",
    "NoWeighting",
    "SampleWeighting",
    "WeightingStrategy",
    "_resolve_weighting",
]
