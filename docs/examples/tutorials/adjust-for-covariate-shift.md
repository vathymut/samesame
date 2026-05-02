# Tutorial: Adjust for covariate shift with importance weights

This tutorial shows how to use contextual RIW weights when testing for adverse shift.
You will estimate membership probabilities from a domain classifier, then apply those
probabilities as weights while testing a separate harmfulness score stream.

**By the end, you will be able to:**

- Compute membership probabilities for contextual weighting
- Keep weighting inputs separate from adverse-shift score inputs
- Run a weighted `test_adverse_shift` and compare it to the unweighted result

If you are new to `samesame`, complete
[Detect a distribution shift](detect-distribution-shift.md)
before this tutorial.

## What you need

- Two groups to compare (source and target)
- A classifier for estimating membership probabilities
- A separate harmfulness score per sample for `test_adverse_shift`

---

## Step 1 — Generate membership probabilities

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

membership_prob = cross_val_predict(
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

Do not reuse `membership_prob` as adverse-shift scores. Instead, create or compute a separate
score where larger means worse.

```python
rng = np.random.default_rng(123_456)

# Separate harmfulness score, independent of membership_prob
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

## Step 3 — Run weighted adverse-shift testing

Pass `membership_prob` to weight the test, and pass `source_scores`/`target_scores` as the
adverse-shift signal.

```python
from samesame import test_adverse_shift

pooled_prob = np.concatenate([
    membership_prob[group == 0],
    membership_prob[group == 1],
])

unweighted = test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    rng=np.random.default_rng(123_456),
)

weighted = test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    membership_prob=pooled_prob,
    mode="source",
    alpha_blend=0.5,
    rng=np.random.default_rng(123_456),
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

- Keep score streams separate: membership probabilities are for weighting; adverse-shift
  scores should come from a harmfulness signal such as risk, error, or low confidence.
- `alpha_blend=0.5` is a practical default.
- Use `mode="both"` when both source and target contain low-overlap outliers.
- For the rationale behind RIW and mode selection, see
    [Why importance weights stabilise shift detection](../../explanation/importance-weights-rationale.md).
