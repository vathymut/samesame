"""Importance weights from domain-classifier probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

_CLIP = 1e-6


class ReweightMode(StrEnum):
    """Which group(s) to reweight toward common support.

    Attributes
    ----------
    SOURCE : ReweightMode
        Reweight source samples to match target.
    TARGET : ReweightMode
        Reweight target samples to match source.
    BOTH : ReweightMode
        Reweight both groups (common-support comparison).
    """

    SOURCE = "source"
    TARGET = "target"
    BOTH = "both"


def _coerce_reweight(value: ReweightMode | str) -> ReweightMode:
    try:
        return ReweightMode(value)
    except ValueError:
        raise ValueError("reweight must be one of 'source', 'target', 'both'.") from None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_prob_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Validate probabilities are finite 1-D arrays in [0, 1] and non-empty."""
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if probs.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(probs)):
        raise ValueError(
            f"{name} domain probabilities must contain only finite values (no NaN or inf)."
        )
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(
            f"{name} domain probabilities must be in the closed interval [0, 1]."
        )
    return probs


def _as_weight_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    w = np.asarray(values, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf).")
    if np.any(w < 0):
        raise ValueError(f"{name} must not contain negative values.")
    if w.sum() == 0:
        raise ValueError(f"{name} must not be all zero.")
    return w


def _normalize(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scale weights to sum to len(weights)."""
    total = float(weights.sum())
    # _as_weight_array already guards against zero-sum, but keep safe.
    if total == 0:
        raise ValueError("weights must not be all zero.")
    return weights / total * len(weights)


# ---------------------------------------------------------------------------
# public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveSampleSize:
    """Kish's effective sample size per group.

    Attributes
    ----------
    source : float
        ESS for the source weights.
    target : float
        ESS for the target weights.
    """

    source: float
    target: float


@dataclass(frozen=True, repr=False)
class ImportanceWeights:
    """Importance weights for source and target.

    Attributes
    ----------
    source : NDArray[np.float64]
        Weights for source samples, normalised to sum to ``len(source)``.
    target : NDArray[np.float64]
        Weights for target samples, normalised to sum to ``len(target)``.

    Notes
    -----
    Weights are normalised on construction so each group sums to its size.
    """

    source: NDArray[np.float64]
    target: NDArray[np.float64]

    def __post_init__(self) -> None:
        src = _normalize(_as_weight_array(self.source, name="weights.source"))
        tgt = _normalize(_as_weight_array(self.target, name="weights.target"))
        object.__setattr__(self, "source", src)
        object.__setattr__(self, "target", tgt)

    def __repr__(self) -> str:
        def _render(a: NDArray[np.float64]) -> str:
            return np.array2string(a, threshold=8, edgeitems=2)

        return f"{type(self).__name__}(source={_render(self.source)}, target={_render(self.target)})"

    def effective_sample_size(self) -> EffectiveSampleSize:
        """Compute Kish's effective sample size.

        ESS quantifies weight concentration: ``(sum w)^2 / sum w^2``.
        Uniform weights give ``ESS == n``; concentrated weights approach 1.

        Returns
        -------
        EffectiveSampleSize
            ESS per group (``.source``, ``.target``).

        References
        ----------
        Kish, L. (1965). Survey Sampling. John Wiley & Sons.

        Examples
        --------
        >>> import numpy as np
        >>> from samesame.weights import domain_weights
        >>> w = domain_weights(source=np.array([0.25, 0.4]), target=np.array([0.6, 0.75]))
        >>> ess = w.effective_sample_size()
        >>> round(ess.source, 4)
        1.8989
        """
        src = float(self.source.sum() ** 2 / (self.source**2).sum())
        tgt = float(self.target.sum() ** 2 / (self.target**2).sum())
        return EffectiveSampleSize(source=src, target=tgt)


# ---------------------------------------------------------------------------
# public: domain_weights
# ---------------------------------------------------------------------------


def domain_weights(
    *,
    source: ArrayLike,
    target: ArrayLike,
    reweight: ReweightMode | str = ReweightMode.BOTH,
    shrinkage: float = 0.5,
) -> ImportanceWeights:
    """Build RIW weights from domain probabilities.

    ``source`` and ``target`` are ``P(target | x)`` for each group.
    The prior ratio ``n_source / n_target`` is inferred from lengths.

    Parameters
    ----------
    source : ArrayLike
        Domain probabilities for source observations in [0, 1]. Clipped to
        ``[1e-6, 1 - 1e-6]`` before weighting.
    target : ArrayLike
        Domain probabilities for target observations in [0, 1].
    reweight : {'source', 'target', 'both'} or ReweightMode, optional
        Which group(s) to reweight. Default ``'both'``. Accepts a plain
        string or :class:`ReweightMode`.
    shrinkage : float, optional
        RIW shrinkage in [0, 1]. ``0`` = plain density ratio, ``1`` = uniform.
        Default ``0.5``.

    Returns
    -------
    ImportanceWeights
        Normalised weights per group. Inactive groups receive weight 1.

    Raises
    ------
    ValueError
        If probabilities are outside [0, 1], empty, or non-finite; if
        ``shrinkage`` is outside [0, 1]; or if ``reweight`` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import domain_weights
    >>> source = np.array([0.25, 0.4])
    >>> target = np.array([0.6, 0.75])
    >>> w = domain_weights(source=source, target=target)
    >>> np.round(w.source, 4)
    array([0.7692, 1.2308])
    """
    reweight_enum = _coerce_reweight(reweight)
    source_p = _as_prob_vector(source, name="source")
    target_p = _as_prob_vector(target, name="target")

    source_p = np.clip(source_p, _CLIP, 1.0 - _CLIP)
    target_p = np.clip(target_p, _CLIP, 1.0 - _CLIP)

    lam = float(shrinkage)
    if not np.isfinite(lam) or lam < 0.0 or lam > 1.0:
        raise ValueError("shrinkage must be in [0, 1] and finite.")

    n_source, n_target = len(source_p), len(target_p)
    prior_ratio = n_source / n_target

    # density ratio r = p/(1-p) * prior_ratio
    source_r = (source_p / (1.0 - source_p)) * prior_ratio
    target_r = (target_p / (1.0 - target_p)) * prior_ratio

    # RIW formulas (Yamada et al. 2013)
    # source: r / ((1-lam) + lam*r), target: 1 / (lam + (1-lam)*r)
    out_source = np.ones(n_source, dtype=np.float64)
    out_target = np.ones(n_target, dtype=np.float64)

    if reweight_enum in (ReweightMode.SOURCE, ReweightMode.BOTH):
        out_source = source_r / ((1.0 - lam) + lam * source_r)
    if reweight_enum in (ReweightMode.TARGET, ReweightMode.BOTH):
        out_target = 1.0 / (lam + (1.0 - lam) * target_r)

    return ImportanceWeights(source=out_source, target=out_target)


__all__ = ["EffectiveSampleSize", "ImportanceWeights", "ReweightMode", "domain_weights"]
