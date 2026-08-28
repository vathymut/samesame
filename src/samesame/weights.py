"""Common support, not just common samples — importance weights for overlap.

An unweighted comparison describes the full source and target samples,
including regions seen by only one group. That is right when those
regions belong in the question you are asking. When the groups barely
overlap, a few observations from the fringes can dominate the statistic
even though the data say little about the other group's behaviour there.

Weighting changes the question to **common support** — the regions of
feature space represented by both groups — without creating information
where the groups do not overlap. It can steady the comparison, but it
also changes the population the test describes.

Use :func:`domain_weights` when you have ``P(target|x)`` from a domain
classifier, or :class:`ImportanceWeights` directly when sample weights
are already available. Start unweighted; reach for weights only when
you have a substantive overlap concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

_CLIP = 1e-6


class ReweightMode(StrEnum):
    """Which group(s) to reweight toward common support.

    Reweighting does not invent information where the groups do not
    overlap — it changes which observations count more. Pick the mode
    that matches where the fringe lives. Pass a member or its plain
    string value to :func:`domain_weights`.

    Attributes
    ----------
    SOURCE : ReweightMode
        Reweight source toward target; target unchanged. Use when source
        has low-overlap observations outside target support.
    TARGET : ReweightMode
        Reweight target toward source; source unchanged. Use when target
        has low-overlap observations outside source support.
    BOTH : ReweightMode
        Reweight both groups toward their mutual support (default).
        Use when both groups have low-overlap regions.

    See Also
    --------
    domain_weights : The function that consumes this choice.
    samesame.weights.ImportanceWeights : What you get back.

    Examples
    --------
    >>> from samesame.weights import ReweightMode
    >>> ReweightMode("both") == ReweightMode.BOTH
    True
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
    """Kish effective sample size — how much information is left after weighting.

    Think of it as the number of equally weighted observations that would
    carry the same information as your unequal weights: ``(sum w)^2 / sum w^2``
    (Kish, 1965). Uniform weights keep every voice — ``ESS == n``. When a
    few observations shout while the rest whisper, ESS slides toward ``1``.

    Compare each value to its ``n``. ``ESS < n/4`` is a friendly warning
    that the weighted result is fragile and largely driven by a few
    observations — not a hard cutoff.

    Attributes
    ----------
    source : float
        ESS for the source weights.
    target : float
        ESS for the target weights.

    See Also
    --------
    ImportanceWeights.effective_sample_size : Compute this from weights.
    samesame.weights.domain_weights : Where shrinkage trades bias for stability.

    References
    ----------
    Kish, L. (1965). Survey Sampling. John Wiley & Sons.
    """

    source: float
    target: float


@dataclass(frozen=True, repr=False)
class ImportanceWeights:
    """Validated, ready-to-use importance weights for source and target.

    Bring your own sample weights, or let :func:`domain_weights` estimate
    them from domain probabilities ``P(target|x)``. Either way, this class
    validates, normalizes, and carries them to the test.

    Reweighting changes which observations count more; it does not change
    nominal group sizes in the permutation test — each group's weights are
    normalized to sum to that group's size, so the labels still permute
    over ``n_source + n_target`` slots.

    Attributes
    ----------
    source : NDArray[np.float64]
        Weights for source observations, normalized to sum to ``len(source)``.
        Inactive groups (per :class:`ReweightMode`) stay at ``1``.
    target : NDArray[np.float64]
        Weights for target observations, normalized to sum to ``len(target)``.
        Inactive groups stay at ``1``.

    Notes
    -----
    On construction, inputs are coerced to finite one-dimensional float
    arrays, checked for non-negativity, and normalized per group. An
    inactive group keeps weight ``1`` for every observation.

    See Also
    --------
    domain_weights : Estimate weights from ``P(target|x)``.
    EffectiveSampleSize : Diagnose weight concentration via
        :meth:`effective_sample_size`.
    samesame.shift.test_shift : The tests that consume these weights.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import ImportanceWeights
    >>> w = ImportanceWeights(source=np.array([0.5, 1.5]), target=np.array([1.0, 1.0]))
    >>> float(w.source.sum()), float(w.target.sum())
    (2.0, 2.0)
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
        """How much independent information remains after weighting.

        Returns Kish's ``(sum w)^2 / sum w^2`` per group. Uniform weights
        give ``ESS == n``; concentrated weights where a handful dominate
        push ESS toward ``1``. A low ESS warns that the weighted result
        leans on a few observations.

        If ESS stays low even at ``shrinkage=0.5``, the groups may lack
        enough common support for a reliable weighted comparison — consider
        leaving the comparison unweighted.

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
    """Turn domain probabilities into weights that focus on where both groups live.

    Give it separate ``P(target|x)`` arrays for source and target
    observations — the probability that each row belongs to target rather
    than source, from a domain classifier. The prior ratio
    ``n_source / n_target`` is inferred from their lengths, so pass arrays
    aligned to the scores you intend to test. Values are clipped to
    ``[1e-6, 1 - 1e-6]`` before ratios to avoid infinities; clipping
    guards the arithmetic but does not rescue a poorly estimated classifier.

    Use the domain probability to build *weights*; use a separate,
    interpretable score (risk, error, confidence, or outlier score) for
    the harm test. Membership is not outcome quality.

    Parameters
    ----------
    source : ArrayLike
        Domain probabilities ``P(target|x)`` for source observations, each
        in ``[0, 1]``. Estimate out of sample (e.g., ``cross_val_predict``)
        or otherwise honestly.
    target : ArrayLike
        Domain probabilities ``P(target|x)`` for target observations, each
        in ``[0, 1]``. Estimate out of sample.
    reweight : {'source', 'target', 'both'} or ReweightMode, optional
        Which group(s) to adjust toward common support. Default
        ``'both'`` — reweight both toward mutual support. ``'source'``
        reweights source toward target; ``'target'`` does the reverse.
        Accepts a plain string or :class:`ReweightMode`.
    shrinkage : float, optional
        Shrinkage ``λ`` in ``[0, 1]`` blending RIW (Relative Importance
        Weight) toward uniform weights. ``0`` is the plain density ratio
        — strongest correction, highest variance. ``1`` is uniform — no
        correction. Default ``0.5`` is a well-tested middle ground; check
        ESS before lowering.

    Returns
    -------
    ImportanceWeights
        Normalized weights per group, each summing to its sample size.
        Inactive groups keep weight ``1`` for every observation.

    Raises
    ------
    ValueError
        If probabilities are outside ``[0, 1]``, empty, or non-finite; if
        ``shrinkage`` is outside ``[0, 1]`` or non-finite; or if
        ``reweight`` is invalid.

    Notes
    -----
    * Start unweighted. Use weights only when you have a substantive
      overlap concern — weighting changes the population the test
      describes and is not a default correction.
    * Estimate ``P(target|x)`` out of sample and keep it separate from
      the harm score. Domain probability describes membership; it says
      nothing about whether an outcome is good or bad.
    * Call ``.effective_sample_size()`` on the result and compare each
      ESS to its ``n``. ``ESS < n/4`` is a warning that a few
      observations dominate. If ESS stays low at ``shrinkage=0.5``,
      the groups may not have enough common support for a reliable
      weighted comparison — consider leaving the comparison unweighted.

    See Also
    --------
    ImportanceWeights : Container that normalizes and validates weights.
    EffectiveSampleSize : Per-group ESS and the ``n/4`` rule of thumb.
    ReweightMode : ``"source"``, ``"target"``, ``"both"`` in plain language.
    samesame.shift.test_shift : Any-shift test that can consume weights.
    samesame.shift.test_harmful_shift : Directional test that can consume weights.

    Examples
    --------
    >>> import numpy as np
    >>> from samesame.weights import domain_weights
    >>> source = np.array([0.25, 0.4])
    >>> target = np.array([0.6, 0.75])
    >>> w = domain_weights(source=source, target=target)
    >>> np.round(w.source, 4)
    array([0.7692, 1.2308])
    >>> w.effective_sample_size().source < 2.0
    True
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
