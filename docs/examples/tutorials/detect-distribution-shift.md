# Tutorial: Detect any distributional shift

Use this tutorial for your first end-to-end shift test between a reference group and a new group.

By the end, you will know how to:

- turn two datasets into a comparable signal
- keep that signal honest with out-of-sample predictions
- run `ss.test_shift` and interpret the result

The workflow: train a classifier to distinguish **source** from **target**. If its out-of-sample probabilities `P(target | x)` separate the two groups more than chance, the datasets differ.

## What you need

- a source dataset and a target dataset
- any scikit-learn classifier with `predict_proba`
- out-of-sample predictions for that classifier

## Step 1 — Create a simple source and target example

We create a synthetic target group with a slight shift on the first feature.

```python
import numpy as np

rng = np.random.default_rng(12345)

source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))

X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In a real workflow, `source` might be training data and `target` might be production data.

## Step 2 — Estimate how much each observation looks like target

Each observation must be scored by a model that did not train on it. `cross_val_predict` is a good default; for bagged forests you can also use `oob_decision_function_` (see [Monitor predicted credit risk](../credit/monitor-credit-risk.md) for the OOB variant).

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X,
    labels,
    cv=10,
    method="predict_proba",
)[:, 1]
```

`domain_prob` is the domain probability `P(target | x)` — the model's estimate that each observation belongs to target. It is used to *compare* source vs target, not as an outlier score of the monitored model.

--8<-- "snippets/honest-scores.txt"

## Step 3 — Run the shift test

```python
import samesame as ss

source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)

print(f"AUC statistic: {shift.statistic:.3f}")
print(f"p-value:       {shift.pvalue:.4f}")
```

On this example, expect a large AUC and a very small p-value — the deliberately shifted target is easy to separate.

## How to read the result

- **Small p-value (typically ≤ 0.05)** — evidence against the null that source and target are the same.
- **Large p-value** — not enough evidence to say the groups differ.
- **Statistic (ROC AUC)** — `0.5` means no separability; values farther from `0.5` mean stronger separation. `test_shift` is two-sided, so AUC `0.2` also rejects — it just means the classifier's ordering is reversed.

`ss.test_shift` answers only "did anything change?" It does not say whether the change is harmful.

If direction matters, continue to [Test whether the shift is harmful](check-shift-harm.md).

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` for reproducible p-values. Use `n_resamples=999` while exploring and `9999` (default) for the final result.
