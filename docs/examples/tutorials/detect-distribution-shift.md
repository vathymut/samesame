# Tutorial: Detect any distributional shift

Your first end-to-end shift test. You will:

- turn two datasets into a comparable score
- keep that score honest with out-of-sample predictions
- run `ss.test_shift` and read the result

Idea: train a classifier to distinguish **source** from **target**. If its out-of-sample `P(target | x)` separates the groups better than chance, they differ.

## What you need

- a source dataset and a target dataset
- any scikit-learn classifier with `predict_proba`
- out-of-sample predictions for that classifier

## Step 1 — Create a source and target example

We create a synthetic target with a slight shift on the first feature.

```python
import numpy as np

rng = np.random.default_rng(12345)

source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))

X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, `source` might be training data and `target` production data.

## Step 2 — Score each observation out of sample

Each observation must be scored by a model that didn't train on it. `cross_val_predict` is a good default; for bagged forests you can use `oob_decision_function_` (see [Monitor predicted credit risk](../credit/monitor-credit-risk.md) for the OOB variant).

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

`domain_prob` is `P(target | x)` — the model's estimate that each row belongs to target. It is the signal for the *shift* test only, not an outlier score of the monitored model.

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

On this shifted example, expect large AUC and very small p-value — the target is easy to separate.

## How to read the result

- **Small p-value (typically ≤ 0.05)** — evidence source and target differ.
- **Large p-value** — not enough evidence to say they differ.
- **Statistic (ROC AUC)** — `0.5` is chance; farther from `0.5` means stronger separation. `test_shift` is two-sided, so AUC `0.2` also rejects — it just means ordering is reversed.

`ss.test_shift` answers only "did anything change?" It says nothing about harm.

Next: [Test whether the shift is harmful](check-shift-harm.md) when direction matters.

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` for reproducible p-values. Use `n_resamples=999` while exploring and `9999` (default) for the final result.
