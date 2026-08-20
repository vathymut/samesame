"""Public importance-weight seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

WeightingMode = Literal["source", "target", "both"]


def _density_ratio(
    domain_prob: ArrayLike,
    *,
    group_balance: float,
) -> NDArray[np.float64]:
    probs = _as_probability_vector(domain_prob, name="domain_prob")
    if not np.isfinite(group_balance) or group_balance <= 0.0:
        raise ValueError("group_balance must be finite and > 0.")
    return (probs / (1.0 - probs)) * group_balance


def _riw(density_ratio_values: NDArray, *, lam: float) -> NDArray[np.float64]:
    return density_ratio_values / ((1.0 - lam) + lam * density_ratio_values)


def _inverse_riw(density_ratio_values: NDArray, *, lam: float) -> NDArray[np.float64]:
    return 1.0 / (lam + (1.0 - lam) * density_ratio_values)


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


@dataclass(frozen=True)
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

    def _as_sample_weight(self, *, n_source: int, n_target: int) -> NDArray[np.float64]:
        source_weight = _check_group_weight_length(
            self.source,
            expected_size=n_source,
            name="weights.source",
        )
        target_weight = _check_group_weight_length(
            self.target,
            expected_size=n_target,
            name="weights.target",
        )
        return np.concatenate([source_weight, target_weight])

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
        >>> from samesame.weights import from_domain_probabilities
        >>> source_prob = np.array([0.25, 0.4])
        >>> target_prob = np.array([0.6, 0.75])
        >>> weights = from_domain_probabilities(
        ...     source_prob=source_prob,
        ...     target_prob=target_prob,
        ...     mode="both"
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


def _check_group_weight_length(
    sample_weight: NDArray[np.float64],
    *,
    expected_size: int,
    name: str,
) -> NDArray[np.float64]:
    if sample_weight.shape[0] != expected_size:
        raise ValueError(
            f"{name} has wrong length: expected {expected_size}, "
            f"got {sample_weight.shape[0]}."
        )
    return sample_weight


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
    if np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError(
            f"{name} domain probabilities must be in the open interval (0, 1)."
        )
    return probabilities


def _validate_lambda(lambda_: float) -> float:
    lambda_value = float(lambda_)
    if not np.isfinite(lambda_value) or lambda_value < 0.0 or lambda_value > 1.0:
        raise ValueError("lambda_ must be in [0, 1] and finite.")
    return lambda_value


def _validate_mode(mode: str) -> WeightingMode:
    match mode:
        case "source" | "target" | "both":
            return mode
    raise ValueError("mode must be one of 'source', 'target', 'both'.")


def from_domain_probabilities(
    *,
    source_prob: ArrayLike,
    target_prob: ArrayLike,
    mode: WeightingMode = "both",
    lambda_: float = 0.5,
) -> ImportanceWeights:
    """Build Importance weights from Domain probabilities.

    Computes RIW weights from domain probabilities.
    The prior ratio is always inferred from the lengths of ``source_prob``
    and ``target_prob``.

    Parameters
    ----------
    source_prob : NDArray
        Domain probabilities for source samples — probability, output by a
        domain classifier, that each source observation belongs to the target
        group. Must be in the open interval (0, 1).
    target_prob : NDArray
        Domain probabilities for target samples — probability, output by a
        domain classifier, that each target observation belongs to the target
        group. Must be in the open interval (0, 1).
    mode : {'source', 'target', 'both'}, optional
        Importance-weighting mode — controls which group's samples are
        reweighted:

        - ``'source'``: reweight source samples only. Use when
          correcting the source distribution to match target.
        - ``'target'``: reweight target samples only. Use when correcting
          the target distribution to match source.
        - ``'both'``: reweight both groups simultaneously (default). Use
          when both groups contain low-overlap outliers.

    lambda_ : float, optional
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
        group's sample size. Samples not targeted by ``mode`` receive weight 1.

    Raises
    ------
    ValueError
        If any value in ``source_prob`` or ``target_prob`` is outside (0, 1).
    ValueError
        If ``lambda_`` is outside [0, 1].
    ValueError
        If ``mode`` is not one of ``'source'``, ``'target'``, ``'both'``.
    ValueError
        If ``source_prob`` or ``target_prob`` is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import from_domain_probabilities
    >>> source_prob = np.array([0.25, 0.4])
    >>> target_prob = np.array([0.6, 0.75])
    >>> w = from_domain_probabilities(source_prob=source_prob, target_prob=target_prob)
    >>> np.round(w.source, 4)
    array([0.7692, 1.2308])
    >>> np.round(w.target, 4)
    array([1.2308, 0.7692])
    >>> w2 = from_domain_probabilities(source_prob=source_prob, target_prob=target_prob, mode="source")
    >>> np.round(w2.source, 4)
    array([0.7692, 1.2308])
    >>> np.round(w2.target, 4)
    array([1., 1.])
    """
    source_prob = _as_probability_vector(source_prob, name="source_prob")
    target_prob = _as_probability_vector(target_prob, name="target_prob")
    n_source = len(source_prob)
    n_target = len(target_prob)
    lambda_value = _validate_lambda(lambda_)
    validated_mode = _validate_mode(mode)

    group_balance = n_source / n_target
    source_dr = _density_ratio(source_prob, group_balance=group_balance)
    target_dr = _density_ratio(target_prob, group_balance=group_balance)

    out_source = np.ones(n_source, dtype=np.float64)
    out_target = np.ones(n_target, dtype=np.float64)

    if validated_mode in ("source", "both"):
        out_source = _riw(source_dr, lam=lambda_value)

    if validated_mode in ("target", "both"):
        out_target = _inverse_riw(target_dr, lam=lambda_value)

    return ImportanceWeights(source=out_source, target=out_target)


__all__ = [
    "EffectiveSampleSize",
    "ImportanceWeights",
    "WeightingMode",
    "from_domain_probabilities",
]
