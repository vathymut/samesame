# Tutorial: Detect a distribution shift

This tutorial is a guided first run of `test_shift(...)`.
You will generate one score per row, run the test, and interpret the result.

**By the end, you will be able to:**

- Check whether two datasets come from the same distribution
- Create one score per row without leaking training data
- Run `test_shift(...)` and read the result

You can use the same workflow when comparing training vs production data, or one batch vs another.

You do not compare the raw feature table directly. Instead, a classifier turns each row into one
score that reflects how strongly it resembles the target dataset rather than the source dataset.
If those scores separate the groups too well, that is evidence that the datasets differ. This
procedure is a **classifier two-sample test**; statistical significance is assessed via a
permutation test on the group labels.

## What you need

- Two datasets to compare (e.g., training data vs. production data)
- A classifier from scikit-learn
- Classifier outputs for rows it did not train on (explained below)

## Step 1 — Prepare the data

Label one dataset as `0` (source) and the other as `1` (target). Combine them.
`make_classification` is used here just to create a quick synthetic example with two groups:

```python
from sklearn.datasets import make_classification

# X contains features; y is the group label (0 = source, 1 = target)
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

# Each sample is scored by a model that never saw it during training
y_hat = cross_val_predict(
    HistGradientBoostingClassifier(random_state=123_456),
    X,
    y,
    cv=10,
    method="predict_proba",
)[:, 1]  # probability of belonging to group 1 (target)
```

## Step 3 — Run the test

Split those model outputs back into source and target groups, then pass those scores to
`test_shift`. The default statistic is ROC AUC. You can think of it as a separation measure:
0.5 means the classifier cannot tell the groups apart, and 1.0 means it separates them perfectly:

```python
source_scores = y_hat[y == 0]
target_scores = y_hat[y == 1]

shift = test_shift(
    source=source_scores,
    target=target_scores,
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
> [Check whether a shift is harmful](check-shift-harm.md).

## Tips

- **Which option should you use?** Most users can keep the default `roc_auc`. Use `balanced_accuracy`
    or `matthews_corrcoef` only when your model output is already binary 0/1 values.
- **Investigate drivers:** If a shift is detected, inspect your classifier's feature importances
    to find which features are most different between the two groups.
- **Shift detected — now what?** A significant shift result means the distributions differ.
    To check whether that difference is actually *harmful*, continue to
    [Check whether a shift is harmful](check-shift-harm.md).
