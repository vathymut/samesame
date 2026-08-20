# Tutorial: Focus on shared support with importance weights

Use this tutorial when source and target do not cover the same feature space well, and you want
your test to focus on the region where they genuinely overlap.

By the end, you will know how to:

- estimate domain probabilities with a domain classifier
- keep weighting inputs separate from the signal you want to test
- compare unweighted and weighted harmful-shift results

Importance weights help when a plain test is being pulled around by observations that are rare or
irrelevant for the comparison you actually care about.

## What you need

- source and target observations
- a domain classifier for estimating `P(target | x)`
- a separate signal for the harmful-shift test

## Step 1 - Estimate domain probabilities

These probabilities are used only for weighting.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

X, group = make_classification(
    n_samples=200,
    n_features=6,
    n_classes=2,
    random_state=123_456,
)

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=123_456),
    X,
    group,
    cv=10,
    method="predict_proba",
)[:, 1]
```

## Step 2 - Build the signal you actually want to monitor

This should be a separate signal. Do not reuse `domain_prob` as the harmful-shift input.

```python
rng = np.random.default_rng(123_456)

risk_score = (
    0.9 * X[:, 0]
    - 0.6 * X[:, 1]
    + 0.4 * X[:, 2]
    + rng.normal(scale=0.4, size=len(group))
)

source_scores = risk_score[group == 0]
target_scores = risk_score[group == 1]
```

## Step 3 - Build weights and compare the test

```python
import samesame as ss
from samesame.weights import from_domain_probabilities

source_prob = domain_prob[group == 0]
target_prob = domain_prob[group == 1]

weights = from_domain_probabilities(
    source_prob=source_prob,
    target_prob=target_prob,
    mode="source",
    lambda_=0.5,
)

unweighted = ss.detect_harm(
    source_scores,
    target_scores,
    direction=ss.Direction.HIGHER_IS_WORSE,
    random_state=123_456,
)

weighted = ss.detect_harm(
    source_scores,
    target_scores,
    direction=ss.Direction.HIGHER_IS_WORSE,
    weights=weights,
    random_state=123_456,
)

print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
print(f"Weighted   p-value: {weighted.pvalue:.4f}")
```

## How to read the result

- The unweighted test uses the full source and target groups.
- The weighted test puts more emphasis on the region where the two groups overlap.
- If the unweighted result is strong but the weighted result is much weaker, the apparent problem
  may be concentrated in low-overlap regions.
- If both results are strong, the signal persists in common support.

`lambda_=0.5` is a practical default. Use `mode="both"` when both source and target contain
low-overlap outliers.

For the intuition behind the weighting formulas, see
[When importance weights help](../../explanation/importance-weights-rationale.md).
