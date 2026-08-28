# Tutorial: Detect any distributional shift

Learn the full loop in 5 minutes: build a score, keep it honest, run `ss.test_shift`.

**Source** is the reference (e.g., training). **Target** is the evaluation batch (e.g., production). See [Glossary](../../explanation/glossary.md#source-and-target).

Idea: train a classifier to distinguish source from target. If its out-of-sample `P(target | x)` separates the groups better than chance, they differ.

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
# → AUC statistic: 0.611
print(f"p-value:       {shift.pvalue:.4f}")
# → p-value:       0.0002
```

On this shifted example, expect AUC ≈ 0.61 and `p < 0.001` — the target separates from source.

## How to read the result

--8<-- "snippets/pvalue-guidance.txt"

- **Statistic (ROC AUC)** — `0.5` is chance; farther from `0.5` means stronger separation. `test_shift` is two-sided, so AUC `0.2` also rejects — ordering is reversed.

`ss.test_shift` answers only "did anything change?" It says nothing about harm. For harm, see [Test whether the shift is harmful](check-shift-harm.md).

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` for reproducible p-values.

    --8<-- "snippets/n-resamples.txt"
