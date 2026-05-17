# Tutorial: Adjust for covariate shift with importance weights

This tutorial shows how to use importance weights when testing for harmful shift.
You will estimate domain probabilities from a domain classifier, then apply those
probabilities as weights while testing a separate harmfulness score stream.

**By the end, you will be able to:**

- Compute domain probabilities for importance weighting
- Keep weighting inputs separate from harmful-shift score inputs
- Run a weighted `ss.shift.detect_harm` and compare it to the unweighted result

If you are new to `samesame`, complete
[Detect a distribution shift](detect-distribution-shift.md)
before this tutorial.

## What you need

- Two groups to compare (source and target)
- A domain classifier for estimating domain probabilities
- A separate harmfulness score per sample for `ss.shift.detect_harm`

---

## Step 1 — Generate domain probabilities

This matches the detect-distribution-shift setup: train a classifier to distinguish source
from target and use out-of-sample probabilities.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

# X contains features; group is 0 (source) or 1 (target)
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
)[:, 1]  # P(target | x)
```

These probabilities are for contextual weighting only.

---

## Step 2 — Build a separate harmfulness score stream

Do not reuse `domain_prob` as harmful-shift scores. Instead, create or compute a separate
score where larger means worse.

```python
rng = np.random.default_rng(123_456)

# Separate harmfulness score, independent of domain_prob
risk_score = (
    0.9 * X[:, 0]
    - 0.6 * X[:, 1]
    + 0.4 * X[:, 2]
    + rng.normal(scale=0.4, size=len(group))
)

source_scores = risk_score[group == 0]
target_scores = risk_score[group == 1]
```

---

## Step 3 — Run weighted harmful-shift testing

Split `domain_prob` by group label to get separate source and target arrays, build
weights with `from_domain_probabilities`, then pass them to `ss.shift.detect_harm`.

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

unweighted = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
    random_state=123_456,
)

weighted = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
    weights=weights,
    random_state=123_456,
)

print(f"Unweighted statistic: {unweighted.statistic:.4f}, p-value: {unweighted.pvalue:.4f}")
print(f"Weighted   statistic: {weighted.statistic:.4f}, p-value: {weighted.pvalue:.4f}")
```

---

## Reading the results

| Result | Interpretation |
|--------|---------------|
| Unweighted adverse shift | Harm signal across the full source and target groups, including outliers. |
| Weighted adverse shift | Harm signal focused on common support after contextual weighting. |

If unweighted is significant but weighted is not, adverse shift may be concentrated in
low-overlap regions. If both are significant, the adverse shift persists in common support.

---

## Tips

- Keep score streams separate: domain probabilities are for weighting; harmful-shift
  scores should come from a harmfulness signal such as risk, error, or low confidence.
- `lambda_=0.5` is a practical default.
- Use `mode="both"` when both source and target contain low-overlap outliers.
- For the rationale behind RIW and mode selection, see
    [Why importance weights stabilise shift detection](../../explanation/importance-weights-rationale.md).
