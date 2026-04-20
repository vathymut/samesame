# Tutorial: Detect a distribution shift

**What you'll learn:**

- How to check whether two datasets come from the same distribution
- How to get fair, unbiased classifier predictions
- How to run `test_shift(...)` and read the result

**Goal:** Determine whether two datasets come from the same underlying distribution.

This is useful any time you need to compare two groups of data — for example, checking whether
your production data still looks like your training data, or whether this week's batch matches
last week's.

`samesame` assumes you already have out-of-sample scores from a model that tried to separate the
two datasets. If those scores separate the groups too well, the two datasets are probably different.

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
from samesame import test_shift

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

Split the out-of-sample scores back into reference and candidate vectors, then pass those score
vectors to `test_shift`. The default statistic is ROC AUC, so 0.5 means no separation
(groups look the same), 1.0 means perfect separation (groups are clearly different):

```python
reference_scores = y_hat[y == 0]
candidate_scores = y_hat[y == 1]

shift = test_shift(
    reference=reference_scores,
    candidate=candidate_scores,
)
print(f"  statistic (AUC): {shift.statistic:.2f}")
print(f"  p-value:         {shift.pvalue:.4f}")
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

> **Important:** `test_shift` tells you *whether* distributions differ, not *how bad* the difference is
> or whether it will hurt your model. For that, see
> [Check whether a shift is harmful](noninferiority.md).

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

shift_oob = test_shift(
    reference=y_oob[y == 0],
    candidate=y_oob[y == 1],
)
print(f"  statistic (AUC): {shift_oob.statistic:.2f}")
print(f"  p-value:         {shift_oob.pvalue:.4f}")
```

**Output:**

```text
    statistic (AUC): 0.94
    p-value:         0.0002
```

Both approaches give the same conclusion here. Use OOB when you already have a Random Forest;
use cross-fitting for any other classifier.

## Advanced options

The primary API is intentionally minimal. When you need weights, custom resampling depth,
or raw null distributions, use `samesame.advanced.test_shift(...)`:

```python
from samesame import advanced

weights = np.ones_like(y_hat)

shift_weighted = advanced.test_shift(
    reference=y_hat[y == 0],
    candidate=y_hat[y == 1],
    sample_weight=weights,
    alternative="greater",  # one-sided alternative
    n_resamples=4999,        # trade precision vs. runtime
)
```

Practical defaults:

- `test_shift(...)` uses `statistic="roc_auc"` by default
- the primary API uses statistically rigorous defaults and hides tuning knobs
- `sample_weight` is an advanced feature only

## Tips

- **AUC vs. binary statistics:** Use the default AUC when your classifier outputs probabilities.
    Use `balanced_accuracy` or `matthews_corrcoef` only when your score vectors are already binary.
- **Investigate drivers:** If a shift is detected, inspect your classifier's feature importances
    to find which features are most different between the two groups.
- **Shift detected — now what?** A significant shift result means the distributions differ.
    To check whether that difference is actually *harmful*, continue to
    [Check whether a shift is harmful](noninferiority.md).
