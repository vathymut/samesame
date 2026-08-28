# Tutorial: Adjust for covariate shift with importance weights

Focus the test on **common support** — the region where [source and target](../../explanation/glossary.md#source-and-target) overlap. Use this when a plain test is driven by low-overlap points the other group rarely contains.

You will:

- estimate domain probabilities `P(target | x)` with a domain classifier (for weighting only)
- keep weighting inputs separate from the harm score
- compare unweighted and weighted harmful-shift results

!!! info "Prerequisites"
    - [Test whether the shift is harmful](check-shift-harm.md) — `worse` and harm statistic.
    - [Detect any distributional shift](detect-distribution-shift.md) — domain classifier and honest scores.

## What you need

- source and target observations
- a domain classifier for `P(target | x)`
- a *separate* harm score (don't reuse `P(target | x)`)

## Step 1 — Estimate domain probabilities

These probabilities are for weighting only.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

X, group = make_classification(
    n_samples=200,
    n_features=6,
    n_classes=2,
    random_state=12345,
)

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X,
    group,
    cv=10,
    method="predict_proba",
)[:, 1]
```

The pooled `domain_prob` is split into `source` and `target` below; the prior ratio `n_source / n_target` is inferred from group sizes.

## Step 2 — Build the harm score

This must be a separate signal. Don't reuse `domain_prob`.

```python
rng = np.random.default_rng(12345)

risk_score = (
    0.9 * X[:, 0]
    - 0.6 * X[:, 1]
    + 0.4 * X[:, 2]
    + rng.normal(scale=0.4, size=len(group))
)

source_scores = risk_score[group == 0]
target_scores = risk_score[group == 1]
```

--8<-- "snippets/honest-scores-ref.txt"

## Step 3 — Build weights and compare

```python
import samesame as ss

source_prob = domain_prob[group == 0]
target_prob = domain_prob[group == 1]

weights = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",  # or ss.ReweightMode.SOURCE
    shrinkage=0.5,
)

rng = np.random.default_rng(12345)
unweighted = ss.test_harmful_shift(
    source=source_scores, target=target_scores, worse="higher", rng=rng,
)

rng = np.random.default_rng(12345)
weighted = ss.test_harmful_shift(
    source=source_scores, target=target_scores, worse="higher", weights=weights, rng=rng,
)

print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
print(f"Weighted   p-value: {weighted.pvalue:.4f}")
```

--8<-- "snippets/clipping-note.txt"

## How to read the result

--8<-- "snippets/pvalue-guidance.txt"

- **Unweighted** — every observation at full strength.
- **Weighted** — emphasizes overlap (here: source reweighted toward target).
- If a strong unweighted signal weakens after weighting, the shift was concentrated in low-overlap regions.
- If both stay strong, the signal persists on common support.

??? tip "Keep `worse` consistent"
    Use the same `worse` for unweighted and weighted calls so you compare weighting, not direction.

For `shrinkage` and `reweight` choices, see [When importance weights help](../../explanation/importance-weights-rationale.md). For `shrinkage` values:

--8<-- "snippets/shrinkage-table.txt"

Check [ESS](../weighting/diagnose-weight-concentration.md) before lowering. For a HELOC example, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).
