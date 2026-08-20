# Tutorial: Detect whether two datasets differ

Use this tutorial when you want a first end-to-end shift test between a reference dataset and a
new dataset.

By the end, you will know how to:

- turn two datasets into a comparison signal
- keep that signal honest with out-of-sample predictions
- run `ss.detect_shift(...)` and interpret the result

The idea is straightforward: train a classifier to tell **source** from **target** apart. If its
out-of-sample probabilities separate the two groups more than chance, the datasets differ.

## What you need

- a source dataset and a target dataset
- any scikit-learn classifier with `predict_proba`
- out-of-sample predictions for that classifier

## Step 1 - Create a simple source and target example

Here we make a synthetic target group that is slightly shifted away from the source group.

```python
import numpy as np

rng = np.random.default_rng(123_456)

source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))

X = np.vstack([source, target])
group = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In a real workflow, `source` might be training data and `target` might be production data.

## Step 2 - Estimate how much each observation looks like target

Each observation must be scored by a model that did not train on it. `cross_val_predict(...)` is a
good default because it handles that for you.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

prob_target = cross_val_predict(
    HistGradientBoostingClassifier(random_state=123_456),
    X,
    group,
    cv=10,
    method="predict_proba",
)[:, 1]
```

`prob_target` is the model's estimated probability that each observation belongs to the target
group.

## Step 3 - Run the shift test

```python
import samesame as ss

source_scores = prob_target[group == 0]
target_scores = prob_target[group == 1]

shift = ss.detect_shift(source_scores, target_scores)

print(f"AUC statistic: {shift.statistic:.3f}")
print(f"p-value:       {shift.pvalue:.4f}")
```

On this example, you should see a large AUC and a very small p-value, which is what we expect from
a deliberately shifted target group.

## How to read the result

- A small p-value means the target group looks different from the source group.
- A large p-value means there is not enough evidence to say the groups differ.
- The default statistic is ROC AUC: `0.5` means the classifier cannot separate the groups, and
  larger values mean stronger separation.

`ss.detect_shift(...)` answers only the question "did anything change?" It does not tell you
whether the change is worse for your application.

If direction matters, continue to
[Check whether a change points in a worse direction](check-shift-harm.md).
