# Tutorial: Adjust for covariate shift with importance weights

Use this tutorial when source and target do not cover the same feature space well and you want your test to focus on the region where they overlap.

By the end, you will know how to:

- estimate domain probabilities `P(target | x)` with a domain classifier
- keep weighting inputs separate from the signal you want to test
- compare unweighted and weighted harmful-shift results

Importance weights help when a plain test is driven by observations that sit where the other group almost never goes. Weighting narrows the comparison to common support.

## What you need

- source and target observations
- a domain classifier for estimating `P(target | x)`
- a *separate* score for the harmful-shift test

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

`domain_prob` pooled is split into `source_prob` and `target_prob` below. The prior ratio `n_source / n_target` is inferred from group sizes.

## Step 2 — Build the outcome score to monitor

This must be a separate signal. Do not reuse `domain_prob` as the harmful-shift input — that would test the domain classifier against itself.

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

--8<-- "snippets/honest-scores.txt"

## Step 3 — Build weights and compare the test

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

- **Unweighted** — uses every observation at full strength.
- **Weighted** — emphasizes the region where the two groups overlap.
- If a strong unweighted result weakens substantially after weighting, the apparent shift was concentrated in low-overlap regions.
- If both stay strong, the signal persists on common support.

`shrinkage=0.5` is the default — it balances correction against variance (see [When importance weights help](../../explanation/importance-weights-rationale.md)). Use `reweight="both"` (or `ss.ReweightMode.BOTH`) when both groups contain low-overlap observations.

??? tip "Keep `worse` consistent"
    Use the same `worse` for unweighted and weighted calls so the comparison is about weighting, not direction.

For how to pick `reweight` and `shrinkage`, see [When importance weights help](../../explanation/importance-weights-rationale.md).
For a worked HELOC example, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).
