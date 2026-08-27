# Tutorial: Detect whether two datasets differ

Use this tutorial when you want a first end-to-end shift test between a reference dataset and a
new dataset.

By the end, you will know how to:

- turn two datasets into a comparison signal
- keep that signal honest with out-of-sample predictions
- run `ss.test_shift(...)` and interpret the result

The workflow is: train a classifier to distinguish **source** from **target**. If its
out-of-sample probabilities separate the two groups more than chance, the datasets differ.

## What you need

- a source dataset and a target dataset
- any scikit-learn classifier with `predict_proba`
- out-of-sample predictions for that classifier

## Step 1 - Create a simple source and target example

We create a synthetic target group with a slight shift from the source group.

```python
import numpy as np

rng = np.random.default_rng(123_456)

source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))

X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In a real workflow, `source` might be training data and `target` might be production data.

## Step 2 - Estimate how much each observation looks like target

Each observation must be scored by a model that did not train on it. `cross_val_predict(...)` is a
good default because it handles that for you.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_score = cross_val_predict(
    HistGradientBoostingClassifier(random_state=123_456),
    X,
    labels,
    cv=10,
    method="predict_proba",
)[:, 1]
```

`domain_score` is the model's estimated probability that each observation belongs to the target
group. It is a domain score used to compare the source and target distributions, not an outlier score
from the monitored model.

## Step 3 - Run the shift test

```python
import samesame as ss

source_scores = domain_score[labels == 0]
target_scores = domain_score[labels == 1]

shift = ss.test_shift(source_scores, target_scores)

print(f"AUC statistic: {shift.statistic:.3f}")
print(f"p-value:       {shift.pvalue:.4f}")
```

On this example, you should see a large AUC and a very small p-value, which is what we expect from
a deliberately shifted target group.

## How to read the result

- A small p-value means the target group looks different from the source group.
- A large p-value means there is not enough evidence to say the groups differ.
- The default statistic is ROC AUC: a value near `0.5` indicates little separability. Values farther
  from `0.5` indicate stronger separation; values below `0.5` mean the classifier's ordering is
  reversed.

`ss.test_shift(...)` answers only the question "did anything change?" It does not tell you
whether the change is worse for your application.

If direction matters, continue to
[Check whether target shifted toward worse outcomes](check-shift-harm.md).
