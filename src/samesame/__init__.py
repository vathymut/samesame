"""samesame — did the target shift? Did it get worse?

Focus monitoring on one meaningful score per observation — predicted risk,
prediction error, confidence, or an outlier score — and compare it between
**source** (the reference, such as training data or a past deployment) and
**target** (the current deployment).

Two questions, two tests. :mod:`samesame.shift` separates them so they are
not conflated:

* ``test_shift`` — broad, two-sided screen for any distributional change
  (ROC AUC).
* ``test_harmful_shift(..., worse=...)`` — focused, one-sided test for
  movement toward the tail you declare harmful (weighted AUC).

When source and target barely overlap, :mod:`samesame.weights` reframes the
comparison around common support via :func:`samesame.weights.domain_weights`
or an explicit :class:`samesame.weights.ImportanceWeights`.

Workflow: choose a score (generate it out of sample if it comes from a fitted
model) → ask whether anything changed → ask whether it got worse → reweight
only if overlap is a real concern. A small p-value is evidence against label
exchangeability, not a measure of business impact or causality.

Public surface is :mod:`samesame.shift` and :mod:`samesame.weights`; start
with :doc:`Get started <examples/tutorials/get-started>` or
:doc:`Monitor a credit model <examples/credit/monitor-credit>`.
"""

from . import shift, weights
from .shift import Worse, test_harmful_shift, test_shift
from .weights import (
    EffectiveSampleSize,
    ImportanceWeights,
    ReweightMode,
    domain_weights,
)

__all__ = [
    "EffectiveSampleSize",
    "ImportanceWeights",
    "ReweightMode",
    "Worse",
    "domain_weights",
    "shift",
    "test_harmful_shift",
    "test_shift",
    "weights",
]
