# Tutorial: Detect a distribution shift

This tutorial is a guided first run of `test_shift(...)`.
You will generate one score per row, run the test, and interpret the result.

**By the end, you will be able to:**

- How to check whether two datasets come from the same distribution
- How to create one score per row without leaking training data
- How to run `test_shift(...)` and read the result

You can use the same workflow when comparing training vs production data, or one batch vs another.

You do not compare the raw feature table directly. Instead, a classifier turns each row into one
score that reflects how strongly it resembles the new dataset rather than the reference dataset.
If those scores separate the groups too well, that is evidence that the datasets differ. This
procedure is a **classifier two-sample test**; statistical significance is assessed via a
permutation test on the group labels.

## What you need

- Two datasets to compare (e.g., training data vs. production data)
- A classifier from scikit-learn
- Classifier outputs for rows it did not train on (explained below)

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

## Step 2 — Create one score per row

This step is important. If you train a classifier on the full dataset and then evaluate the same
rows, the classifier can appear artificially strong because it is being tested on data it has already seen.
For a valid comparison, each row must be evaluated by a model that did *not* train on it. These values are
often called **out-of-sample predictions**.

**Recommended: use `cross_val_predict`**

`cross_val_predict` splits the data into folds. Each row is then evaluated by a model trained
on the remaining folds. This is the safest default for most users:

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

Split those model outputs back into reference and candidate groups, then pass those scores to
`test_shift`. The default statistic is ROC AUC. You can think of it as a separation measure:
0.5 means the classifier cannot tell the groups apart, and 1.0 means it separates them perfectly:

```python
reference_values = y_hat[y == 0]
candidate_values = y_hat[y == 1]

shift = test_shift(
    reference=reference_values,
    candidate=candidate_values,
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
which is strong evidence against the null hypothesis of no distributional difference.

> **Important:** `test_shift` tells you *whether* distributions differ, not *how bad* the difference is
> or whether it will hurt your model. For that, see
> [Check whether a shift is harmful](/examples/tutorials/check-shift-harm.md).

## Alternative: out-of-bag (OOB) predictions

If you are using a Random Forest or another ensemble model, you can use its built-in
**out-of-bag (OOB)** predictions instead of running `cross_val_predict` explicitly.
This is the Random Forest version of the same idea: each row is evaluated only by trees that did not
train on it:

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
use cross-fitting for other classifiers.

## Want more control?

If you need sample weights, more resamples, or the full null distribution, use
`samesame.advanced.test_shift(...)`. Most readers can skip that on a first pass and return to it later.

## Tips

- **Which option should you use?** Most users can keep the default `roc_auc`. Use `balanced_accuracy`
    or `matthews_corrcoef` only when your model output is already binary 0/1 values.
- **Investigate drivers:** If a shift is detected, inspect your classifier's feature importances
    to find which features are most different between the two groups.
- **Shift detected — now what?** A significant shift result means the distributions differ.
    To check whether that difference is actually *harmful*, continue to
    [Check whether a shift is harmful](/examples/tutorials/check-shift-harm.md).
