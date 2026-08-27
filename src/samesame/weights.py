"""Public importance-weight seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

ReweightMode = Literal["source", "target", "both"]
_PROBABILITY_CLIP = 1e-6


def _density_ratio(
    domain_prob: NDArray[np.float64],
    *,
    prior_ratio: float,
) -> NDArray[np.float64]:
    """Density ratio ``p(target)/p(source)`` implied by domain probabilities."""
    return (domain_prob / (1.0 - domain_prob)) * prior_ratio


def _riw(density_ratio_values: NDArray, *, shrinkage: float) -> NDArray[np.float64]:
    return density_ratio_values / ((1.0 - shrinkage) + shrinkage * density_ratio_values)


def _inverse_riw(
    density_ratio_values: NDArray, *, shrinkage: float
) -> NDArray[np.float64]:
    return 1.0 / (shrinkage + (1.0 - shrinkage) * density_ratio_values)


@dataclass(frozen=True)
class EffectiveSampleSize:
    """Kish's effective sample size for the source and target groups.

    Attributes
    ----------
    source : float
        Effective sample size for the source group.
    target : float
        Effective sample size for the target group.
    """

    source: float
    target: float


@dataclass(frozen=True, repr=False)
class ImportanceWeights:
    """Importance weights for Source and Target groups.

    Attributes
    ----------
    source : NDArray[np.float64]
        Importance weights for source samples, normalized to sum to
        ``len(source)``.
    target : NDArray[np.float64]
        Importance weights for target samples, normalized to sum to
        ``len(target)``.
    """

    source: NDArray[np.float64]
    target: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _normalize_group_weights(self.source, name="weights.source"),
        )
        object.__setattr__(
            self,
            "target",
            _normalize_group_weights(self.target, name="weights.target"),
        )

    def __repr__(self) -> str:
        def render(values: NDArray[np.float64]) -> str:
            return np.array2string(values, threshold=8, edgeitems=2)

        return (
            f"{type(self).__name__}("
            f"source={render(self.source)}, target={render(self.target)})"
        )

    def effective_sample_size(self) -> EffectiveSampleSize:
        """Compute effective sample size for source and target weights.

        Returns Kish's effective sample size (ESS) for each group, quantifying
        the degree of weight concentration. Lower ESS indicates that fewer
        observations dominate the weighted comparison.

        The formula is ``(Σw)² / Σw²``, applied separately to source and target
        weights. For uniform weights, ESS equals the sample size; for highly
        concentrated weights, ESS approaches 1.

        Returns
        -------
        EffectiveSampleSize
            Kish's effective sample size per group, accessible via the
            ``.source`` and ``.target`` attributes.

        References
        ----------
        Kish, L. (1965). Survey Sampling. John Wiley & Sons.

        Examples
        --------
        >>> import numpy as np
        >>> from samesame.weights import domain_weights
        >>> source = np.array([0.25, 0.4])
        >>> target = np.array([0.6, 0.75])
        >>> weights = domain_weights(
        ...     source=source,
        ...     target=target,
        ...     reweight="both"
        ... )
        >>> ess = weights.effective_sample_size()
        >>> round(ess.source, 4)
        1.8989
        >>> round(ess.target, 4)
        1.8989
        """
        source_sum = self.source.sum()
        source_sum_sq = (self.source**2).sum()
        source_ess = float(source_sum**2 / source_sum_sq)

        target_sum = self.target.sum()
        target_sum_sq = (self.target**2).sum()
        target_ess = float(target_sum**2 / target_sum_sq)

        return EffectiveSampleSize(source=source_ess, target=target_ess)


def _as_group_weight_array(
    sample_weight: ArrayLike, *, name: str
) -> NDArray[np.float64]:
    weight = np.asarray(sample_weight, dtype=np.float64)
    if weight.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(weight)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf).")
    if np.any(weight < 0):
        raise ValueError(f"{name} must not contain negative values.")
    total = weight.sum()
    if total == 0:
        raise ValueError(f"{name} must not be all zero.")
    return weight


def _normalize_group_weights(
    sample_weight: ArrayLike, *, name: str
) -> NDArray[np.float64]:
    weight = _as_group_weight_array(sample_weight, name=name)
    return weight / weight.sum() * len(weight)


def _as_probability_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            f"{name} domain probabilities must contain only finite values "
            "(no NaN or inf)."
        )
    if probabilities.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if probabilities.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            f"{name} domain probabilities must be in the closed interval [0, 1]."
        )
    return probabilities


def _validate_shrinkage(shrinkage: float) -> float:
    shrinkage_value = float(shrinkage)
    if (
        not np.isfinite(shrinkage_value)
        or shrinkage_value < 0.0
        or shrinkage_value > 1.0
    ):
        raise ValueError("shrinkage must be in [0, 1] and finite.")
    return shrinkage_value


def _validate_reweight(reweight: str) -> ReweightMode:
    if reweight not in ("source", "target", "both"):
        raise ValueError("reweight must be one of 'source', 'target', 'both'.")
    return cast("ReweightMode", reweight)


def domain_weights(
    *,
    source: ArrayLike,
    target: ArrayLike,
    reweight: ReweightMode = "both",
    shrinkage: float = 0.5,
) -> ImportanceWeights:
    """Build Importance weights from Domain probabilities.

    Computes RIW weights from domain probabilities. ``source`` and ``target``
    are probabilities that each observation belongs to the target group.
    The prior ratio is inferred from their lengths.

    Parameters
    ----------
    source : NDArray
        Domain probabilities for source samples — probability, output by a
        domain classifier, that each source observation belongs to the target
        group. Must be in the closed interval [0, 1]. Values are clipped to
        ``[1e-6, 1 - 1e-6]`` before calculating weights.
    target : NDArray
        Domain probabilities for target samples — probability, output by a
        domain classifier, that each target observation belongs to the target
        group. Must be in the closed interval [0, 1]. Values are clipped to
        ``[1e-6, 1 - 1e-6]`` before calculating weights.
    reweight : {'source', 'target', 'both'}, optional
        Importance-weighting mode — controls which group's samples are
        reweighted:

        - ``'source'``: reweight source samples only. Use when
          correcting the source distribution to match target.
        - ``'target'``: reweight target samples only. Use when correcting
          the target distribution to match source.
        - ``'both'``: reweight both groups simultaneously (default). Use
          when both groups contain low-overlap outliers.

    shrinkage : float, optional
        RIW blending coefficient in [0, 1] controlling the trade-off between
        correction strength and variance stability. ``0.0`` gives plain
        density-ratio weights (maximum correction, highest variance); ``1.0``
        gives uniform weights (no correction). Default ``0.5`` is a balanced
        starting point for most applications.

    Returns
    -------
    ImportanceWeights
        A frozen dataclass with ``.source`` and ``.target`` weight arrays.
        Weights for each active group are normalized so they sum to that
        group's sample size. Samples not targeted by ``reweight`` receive weight 1.

    Raises
    ------
    ValueError
        If any value in ``source`` or ``target`` is outside [0, 1].
    ValueError
        If ``shrinkage`` is outside [0, 1].
    ValueError
        If ``reweight`` is not one of ``'source'``, ``'target'``, ``'both'``.
    ValueError
        If ``source`` or ``target`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import domain_weights
    >>> source = np.array([0.25, 0.4])
    >>> target = np.array([0.6, 0.75])
    >>> w = domain_weights(source=source, target=target)
    >>> np.round(w.source, 4)
    array([0.7692, 1.2308])
    >>> np.round(w.target, 4)
    array([1.2308, 0.7692])
    >>> w2 = domain_weights(source=source, target=target, reweight="source")
    >>> np.round(w2.source, 4)
    array([0.7692, 1.2308])
    >>> np.round(w2.target, 4)
    array([1., 1.])
    """
    source = _as_probability_vector(source, name="source")
    target = _as_probability_vector(target, name="target")
    source = np.clip(source, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
    target = np.clip(target, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
    n_source = len(source)
    n_target = len(target)
    shrinkage_value = _validate_shrinkage(shrinkage)
    validated_reweight = _validate_reweight(reweight)

    prior_ratio = n_source / n_target
    source_dr = _density_ratio(source, prior_ratio=prior_ratio)
    target_dr = _density_ratio(target, prior_ratio=prior_ratio)

    out_source = np.ones(n_source, dtype=np.float64)
    out_target = np.ones(n_target, dtype=np.float64)

    if validated_reweight in ("source", "both"):
        out_source = _riw(source_dr, shrinkage=shrinkage_value)

    if validated_reweight in ("target", "both"):
        out_target = _inverse_riw(target_dr, shrinkage=shrinkage_value)

    return ImportanceWeights(source=out_source, target=out_target)


__all__ = [
    "EffectiveSampleSize",
    "ImportanceWeights",
    "domain_weights",
]
