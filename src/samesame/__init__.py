"""samesame — score-based source-versus-target monitoring.

Reduce each observation to one interpretable score — predicted risk,
prediction error, confidence, or outlier score — and compare its
distribution between **source** (reference: training or past deployment)
and **target** (current deployment). The raw feature space is often too
large to interpret and labels can arrive late; a single score gives each
row one number to monitor.

Two questions, two tests. :mod:`samesame.shift` separates them so they are
not conflated:

* ``test_shift`` — broad, two-sided screen for any distributional change
  (ROC AUC ``∫ TPR dFPR``, ``0.5`` is chance).
* ``test_harmful_shift(..., worse=...)`` — focused, one-sided test for
  movement toward the tail you declare harmful (weighted AUC
  ``∫ TPR·(1−FPR)² dFPR``).

When poor feature overlap is a real concern, :mod:`samesame.weights`
reframes the comparison around common support — the regions represented by
both groups — via :func:`samesame.weights.domain_weights` or an explicit
:class:`samesame.weights.ImportanceWeights`. Weighting changes which
observations count more; it does not create information where groups do not
overlap.

Workflow: (1) choose one score per observation — generate it out of sample
with ``cross_val_predict``, ``oob_decision_function_``, or a held-out set if
it comes from a fitted model; (2) ask whether anything changed
(``test_shift``); (3) ask whether it got worse
(``test_harmful_shift`` with ``worse``); (4) reweight only if poor overlap
is a real concern. A small p-value is evidence against label
exchangeability — not business impact, causality, or the probability the null
is true.

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
