# How to Detect Distribution Shifts

**Goal:** Determine whether two datasets come from the same underlying distribution.

This is useful any time you need to compare two groups of data — for example, checking whether
your production data still looks like your training data, or whether this week's batch matches
last week's.

`samesame` uses a **Classifier Two-Sample Test (CTST)** for this. The core idea is simple:
train a classifier to distinguish between the two datasets. If it can do so reliably (high AUC),
the two datasets are probably different. If it can't do better than random, they are likely the same.

## What you need

- Two datasets to compare (e.g., training data vs. production data)
- A classifier from scikit-learn
- Out-of-sample predictions from that classifier (explained below)

## Step 1 — Prepare the data

Label one dataset as `0` (reference) and the other as `1` (new). Combine them.
`make_classification` is used here just to create a quick synthetic example with two groups:

```python
from sklearn.datasets import make_classification

# X contains the features; y is the group label (0 = reference, 1 = new)
X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_classes=2,
        random_state=123_456,
)
```

## Step 2 — Get out-of-sample predictions

This step is important. If you train a classifier on the full dataset and then score the same
data, the classifier will appear to perform well even if the groups are actually identical —
it simply memorises the training data. To get a fair test, you need **out-of-sample predictions**,
where each sample is scored by a model that was *not* trained on it.

**Recommended: cross-fitting with `cross_val_predict`**

Cross-fitting splits the data into folds and makes predictions on each fold using a model
trained on the remaining folds. This is the most statistically sound approach:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from samesame.ctst import CTST

# Predict each sample's probability using a model that never saw that sample
y_hat = cross_val_predict(
        HistGradientBoostingClassifier(random_state=123_456),
        X,
        y,
        cv=10,
        method="predict_proba",
)[:, 1]  # take the probability of belonging to group 1
```

## Step 3 — Run the test

Pass the group labels and the predictions to `CTST`. The AUC (area under the ROC curve)
measures how well the classifier separates the two groups — 0.5 means no separation
(groups look the same), 1.0 means perfect separation (groups are clearly different):

```python
ctst = CTST(actual=y, predicted=y_hat, metric=roc_auc_score)
print(f"  statistic (AUC): {ctst.statistic:.2f}")
print(f"  p-value:         {ctst.pvalue:.4f}")
```

**Output:**

```text
    statistic (AUC): 0.93
    p-value:         0.0002
```

## Reading the results

| p-value         | What it means                                                  |
|-----------------|----------------------------------------------------------------|
| Small (< 0.05)  | Strong evidence that the two datasets come from different distributions |
| Large (≥ 0.05)  | Not enough evidence to conclude the distributions differ       |

Here, p = 0.0002 is very small — the classifier can easily tell the two groups apart,
which is strong evidence of a distributional difference.

> **Important:** CTST tells you *whether* distributions differ, not *how bad* the difference is
> or whether it will hurt your model. For that, see the [Noninferiority guide](noninferiority.md).

## Alternative: out-of-bag (OOB) predictions

If you are using a Random Forest or another ensemble model, you can use its built-in
**out-of-bag (OOB)** predictions instead of running cross-fitting explicitly.
OOB predictions are generated during training for free — each tree only scores samples
it was not trained on:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        min_samples_leaf=10,
        random_state=123_456,
)
rf.fit(X, y)
y_oob = rf.oob_decision_function_[:, 1]

ctst_oob = CTST(actual=y, predicted=y_oob, metric=roc_auc_score)
print(f"  statistic (AUC): {ctst_oob.statistic:.2f}")
print(f"  p-value:         {ctst_oob.pvalue:.4f}")
```

**Output:**

```text
    statistic (AUC): 0.94
    p-value:         0.0002
```

Both approaches give the same conclusion here. Use OOB when you already have a Random Forest;
use cross-fitting for any other classifier.

## Common CTST options

Use `CTST.from_samples(...)` when you only have two score vectors and do not need per-row weights.
When you need more control, build `CTST(...)` directly:

```python
weights = np.ones_like(y_hat)

ctst_weighted = CTST(
    actual=y,
    predicted=y_hat,
    metric=roc_auc_score,
    sample_weight=weights,
    alternative="greater",  # one-sided alternative
    n_resamples=4999,         # trade precision vs. runtime
)
```

Practical defaults:

- `alternative="two-sided"` is the default and best for generic shift detection
- `n_resamples=9999` is the default and gives stable p-values in most settings
- `sample_weight` is optional; pass it only when some observations should count more than others

## Tips

- **AUC vs. balanced accuracy:** Use AUC when your classifier outputs probabilities. Use balanced
    accuracy when it outputs binary labels. Both work with `CTST`.
- **Investigate drivers:** If a shift is detected, inspect your classifier's feature importances
    to find which features are most different between the two groups.
- **Shift detected — now what?** A significant CTST result means the distributions differ.
    To check whether that difference is actually *harmful*, continue to
    [Noninferiority testing](noninferiority.md).
