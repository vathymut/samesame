"""Shift tests — score-based source-versus-target monitoring.

Compare one meaningful score per observation — predicted risk, prediction
error, confidence, or outlier score — between **source** (reference:
training or past deployment) and **target** (current deployment). The raw
feature space is often too large to interpret and labels can arrive late;
a single score gives each row one number to monitor.

Both tests keep scores — and any importance weights — fixed and shuffle
group labels. A small p-value is evidence against label exchangeability
— not business impact, causality, or the probability the null is true.

Two questions, two tests — use them separately so they are not
conflated:

* :func:`test_shift` — broad, two-sided screen. Can the score
  distinguish source from target at all? Reports ROC AUC (``0.5`` is
  chance).
* :func:`test_harmful_shift` — focused, one-sided tail test. After
  orienting the score so larger means worse, does target put more mass
  beyond thresholds that source rarely exceeds? Reports a weighted AUC
  ``∫ TPR·(1−FPR)² dFPR`` that emphasizes that harmful tail.

Start unweighted. Reach for :mod:`samesame.weights` only when poor
feature overlap is a real concern — weighting reframes the comparison
around common support and changes the population described; it does not
create information where groups do not overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score

from samesame._permutation import Seed, _permutation_test
from samesame._statistics import harmful_shift_statistic
from samesame.weights import ImportanceWeights


class Worse(StrEnum):
    """Polarity that defines which tail is harmful for :func:`test_harmful_shift`.

    Declare ``worse`` from what the score means before looking at
    results — not from whichever direction gives the smaller p-value.
    A plain string ``"higher"`` / ``"lower"`` is accepted wherever this
    enum is; the two forms are interchangeable.

    Attributes
    ----------
    HIGHER : Worse
        Larger scores mean more harm (e.g., predicted risk, prediction
        error, or atypicality outlier score).
    LOWER : Worse
        Smaller scores mean more harm (e.g., confidence via ``LogitGap``;
        lower is worse).

    See Also
    --------
    samesame.shift.test_harmful_shift : The test that consumes this choice.

    Examples
    --------
    >>> from samesame import Worse
    >>> Worse("higher") == Worse.HIGHER
    True
    >>> Worse("lower") == Worse.LOWER
    True
    """

    HIGHER = "higher"
    LOWER = "lower"


def _coerce_worse(value: Worse | str) -> Worse:
    try:
        return Worse(value)
    except ValueError:
        raise ValueError("worse must be either 'higher' or 'lower'.") from None


def _fmt(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float) else repr(v)


@dataclass(frozen=True)
class ShiftResult:
    """Result of :func:`test_shift` — a two-sided permutation result.

    The statistic is ROC AUC — how well the score separates target from
    source (``0.5`` is chance; values farther from ``0.5`` signal stronger
    separation, in either direction). The p-value is evidence against label
    exchangeability — not business impact, causality, an effect size, or
    the probability the null is true.

    Attributes
    ----------
    statistic : float
        Observed ROC AUC, weighted when ``weights`` were supplied.
    pvalue : float
        Two-sided permutation p-value with +1 smoothing (always > 0;
        doubling the smaller tail, capped at 1).
    null_distribution : NDArray[np.float64]
        Null distribution of the statistic (length ``n_resamples``),
        produced by permuting group labels while keeping scores and
        weights fixed.

    See Also
    --------
    test_harmful_shift : When you can name the harmful tail in advance.
    samesame.weights.domain_weights : If poor overlap is a real concern.
    """

    statistic: float
    pvalue: float
    null_distribution: NDArray[np.float64]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"statistic={_fmt(self.statistic)}, pvalue={_fmt(self.pvalue)})"
        )


@dataclass(frozen=True, repr=False)
class HarmfulShiftResult(ShiftResult):
    """Result of :func:`test_harmful_shift` — a one-sided tail result.

    The statistic is a weighted AUC ``∫ TPR·(1−FPR)² dFPR`` that leans
    into thresholds the source rarely exceeds, after orienting the score
    so larger means worse (``worse="lower"`` flips the sign). Read it
    against ``null_distribution`` and the score's own scale. See
    :doc:`How the harm test works <../explanation/harmful-shift-statistic>`
    for the ROC intuition.

    Attributes
    ----------
    statistic : float
        Observed harmful-shift statistic.
    pvalue : float
        One-sided (``greater``) permutation p-value with +1 smoothing
        (always > 0).
    null_distribution : NDArray[np.float64]
        Null distribution of the statistic (length ``n_resamples``),
        produced by permuting group labels while keeping scores and
        weights fixed.
    worse : Worse
        The declared harmful direction that was tested.

    See Also
    --------
    test_shift : Broad screen when any change matters.
    Worse : ``"higher"`` vs ``"lower"`` in plain language.
    """

    worse: Worse

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"statistic={_fmt(self.statistic)}, pvalue={_fmt(self.pvalue)}, "
            f"worse={self.worse.value!r})"
        )


# ---------------------------------------------------------------------------
# internal metrics
# ---------------------------------------------------------------------------


def _auc_metric(
    labels: NDArray[np.int_],
    scores: NDArray[np.float64],
    sample_weight: NDArray[np.float64] | None,
) -> float:
    return float(roc_auc_score(labels, scores, sample_weight=sample_weight))


def _harm_metric_factory(worse: Worse):
    def _metric(
        labels: NDArray[np.int_],
        scores: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None,
    ) -> float:
        polarity = scores if worse == "higher" else -scores
        return harmful_shift_statistic(labels, polarity, sample_weight=sample_weight)

    return _metric


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def test_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    n_resamples: int = 9999,
    rng: Seed = None,
    weights: ImportanceWeights | None = None,
) -> ShiftResult:
    """Broad screen — do source and target scores differ at all?

    Any shift? Start here. Give it one meaningful score per observation —
    predicted risk, prediction error, confidence, or outlier score — and
    it measures separation with ROC AUC, then shuffles labels to see
    whether separation is unusual. A small p-value is evidence that the
    distributions differ — not that the shift is harmful, large, or
    causal.

    Choose the score that answers your monitoring question before testing.
    If the score comes from a fitted model, generate it out of sample with
    ``cross_val_predict``, ``oob_decision_function_``, or a held-out set
    — in-sample scores can make the groups look spuriously separable
    because the scoring model has memorised its inputs.

    Parameters
    ----------
    source : ArrayLike
        Scores for the source (reference) group — e.g., training data or a
        past deployment.
    target : ArrayLike
        Scores for the target (evaluation) group — e.g., the current
        deployment.
    n_resamples : int, optional
        Number of label permutations. Default ``9999``. Use ``999`` while
        exploring and ``19999`` for finer resolution below ``0.001``.
    rng : int | np.random.Generator | np.random.RandomState | None, optional
        Random state for reproducibility. Pass ``np.random.default_rng(12345)``
        or an ``int`` seed. Default ``None``.
    weights : ImportanceWeights | None, optional
        Per-observation importance weights aligned to ``source`` and
        ``target``. Omit to compare the full populations; supply when the
        comparison should focus on common support — weights are normalized
        per group to its sample size (inactive groups stay at ``1``) (see
        :func:`samesame.weights.domain_weights`).

    Returns
    -------
    ShiftResult
        Observed AUC, two-sided p-value, and null distribution. The null is
        formed by permuting group labels while keeping scores and weights
        fixed.

    Notes
    -----
    * The p-value doubles the smaller tail (capped at ``1``) and adds ``+1``
      smoothing so it is never exactly zero — permutation p-values stay
      above zero.
    * Interpret ``statistic`` relative to ``0.5`` (chance; ``0.8`` or
      ``0.2`` both signal strong separation) and ``pvalue`` as evidence
      against exchangeability — not as harm or business impact.
    * For honest p-values, scores from a fitted model must be out of
      sample. In-sample predictions can inflate separation because the
      scoring model has memorised its inputs.

    See Also
    --------
    test_harmful_shift : Directional test when you can declare the harmful tail.
    samesame.weights.domain_weights : Build weights from ``P(target|x)``.
    samesame.weights.ImportanceWeights : Container for per-group weights.

    Examples
    --------
    >>> import numpy as np
    >>> import samesame as ss
    >>> rng = np.random.default_rng(12345)
    >>> source = rng.normal(0, 1, size=300)
    >>> target = rng.normal(0.6, 1, size=300)
    >>> res = ss.test_shift(source, target, rng=rng)
    >>> 0.5 < res.statistic <= 1.0
    True
    >>> res.pvalue < 0.01
    True
    """
    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=_auc_metric,
        alternative="two-sided",
        n_resamples=n_resamples,
        rng=rng,
        weights=weights,
    )
    return ShiftResult(
        statistic=statistic, pvalue=pvalue, null_distribution=null_distribution
    )


def test_harmful_shift(
    source: ArrayLike,
    target: ArrayLike,
    *,
    worse: Worse | str,
    n_resamples: int = 9999,
    rng: Seed = None,
    weights: ImportanceWeights | None = None,
) -> HarmfulShiftResult:
    """Focused check — did target move toward the harmful tail you care about?

    A small ``test_shift`` p-value says *something* changed. This test asks
    the narrower question: after orienting the score so larger means worse
    (``worse="lower"`` flips the sign), does target put more mass beyond
    thresholds the source rarely exceeds? Thresholds the source rarely
    exceeds get more weight, so the test leans into the harmful tail. A
    small p-value is evidence for that directional movement — not for
    arbitrary shift.

    Decide ``worse`` from what the score means before looking at results;
    do not pick the direction that gives the smaller p-value.

    Parameters
    ----------
    source : ArrayLike
        Scores for the source (reference) group — e.g., training data or a
        past deployment.
    target : ArrayLike
        Scores for the target (evaluation) group — e.g., the current
        deployment.
    worse : {'higher', 'lower'} or Worse
        Which tail is harmful. ``"higher"`` when larger scores mean harm
        (e.g., predicted risk, prediction error, atypicality outlier
        score); ``"lower"`` when smaller scores mean harm (e.g.,
        confidence via ``LogitGap``). Accepts a plain string or
        :class:`Worse`.
    n_resamples : int, optional
        Number of label permutations. Default ``9999``. Use ``999`` while
        exploring and ``19999`` for finer resolution below ``0.001``.
    rng : int | np.random.Generator | np.random.RandomState | None, optional
        Random state for reproducibility. Pass ``np.random.default_rng(12345)``
        or an ``int`` seed. Default ``None``.
    weights : ImportanceWeights | None, optional
        Per-observation importance weights aligned to ``source`` and
        ``target``. Omit to compare the full populations; supply when the
        comparison should focus on common support — weights are normalized
        per group to its sample size (inactive groups stay at ``1``) (see
        :func:`samesame.weights.domain_weights`).

    Returns
    -------
    HarmfulShiftResult
        Observed weighted-AUC, one-sided p-value, declared ``worse``, and
        null distribution. The null is formed by permuting group labels
        while keeping scores and weights fixed.

    Notes
    -----
    * One-sided ``greater`` alternative with ``+1`` smoothing (never zero).
    * Compare the statistic to ``null_distribution`` and the score's own
      scale, not to ``0.5``. See :doc:`How the harm test works
      <../explanation/harmful-shift-statistic>` for the ROC intuition and
      the ``∫ TPR·(1−FPR)² dFPR`` form.

    See Also
    --------
    test_shift : Broad, two-sided screen when any change matters.
    samesame.weights.domain_weights : Build weights from ``P(target|x)``.
    Worse : The ``"higher"`` / ``"lower"`` choice in plain language.

    Examples
    --------
    >>> import numpy as np
    >>> import samesame as ss
    >>> rng = np.random.default_rng(12345)
    >>> source = rng.normal(0.20, 0.07, size=300)
    >>> target = rng.normal(0.28, 0.07, size=300)  # higher risk = worse
    >>> res = ss.test_harmful_shift(source, target, worse="higher", rng=rng)
    >>> res.pvalue < 0.05
    True
    """
    worse_enum = _coerce_worse(worse)

    metric = _harm_metric_factory(worse_enum)

    statistic, pvalue, null_distribution = _permutation_test(
        source,
        target,
        metric=metric,
        alternative="greater",
        n_resamples=n_resamples,
        rng=rng,
        weights=weights,
    )
    return HarmfulShiftResult(
        statistic=statistic,
        pvalue=pvalue,
        worse=worse_enum,
        null_distribution=null_distribution,
    )


__all__ = ["HarmfulShiftResult", "ShiftResult", "Worse", "test_harmful_shift", "test_shift"]
