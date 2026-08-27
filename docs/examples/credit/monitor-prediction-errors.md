# How to: Monitor prediction errors once labels arrive

Use this guide when you have ground-truth labels for both groups and want a direct answer about
whether the model is performing worse.

When labels are available, prediction error is often the cleanest signal you can compare.

## Why this signal works well

Prediction errors turn model quality into a numeric signal:

- **Brier score** measures squared error on the predicted probability
- **Log-loss** penalizes confident mistakes more heavily

For both, larger values mean worse predictions, so they work naturally with
`ss.detect_harmful_shift(...)`.

## Setup

This guide uses the HELOC dataset again, but with a stratified random split. Unlike the credit-risk
guide, the two groups here come from the same overall population.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import samesame as ss

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

y_binary = (y == "Bad").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_binary,
    test_size=0.30,
    stratify=y_binary,
    rng=np.random.default_rng(12345),
)
```

## Step 1 - Fit the model and get honest predictions

Use out-of-bag predictions for training so the training-side errors are not artificially optimistic.

```python
rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    rng=np.random.default_rng(12345),
    min_samples_leaf=10,
)
rf.fit(X_train, y_train)

train_prob = rf.oob_decision_function_[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]
```

## Step 2 - Turn predictions into error signals

```python
brier_train = (y_train - train_prob) ** 2
brier_test = (y_test - test_prob) ** 2

eps = 1e-10
train_prob_clipped = np.clip(train_prob, eps, 1 - eps)
test_prob_clipped = np.clip(test_prob, eps, 1 - eps)

logloss_train = -(
    y_train * np.log(train_prob_clipped)
    + (1 - y_train) * np.log(1 - train_prob_clipped)
)
logloss_test = -(
    y_test * np.log(test_prob_clipped)
    + (1 - y_test) * np.log(1 - test_prob_clipped)
)
```

## Step 3 - Test whether errors are worse on the test set

```python
harm_brier = ss.detect_harmful_shift(
    source=brier_train,
    target=brier_test,
    higher_is_worse=True,
)

harm_logloss = ss.detect_harmful_shift(
    source=logloss_train,
    target=logloss_test,
    higher_is_worse=True,
)

print(f"Brier p-value:   {harm_brier.pvalue:.4f}")
print(f"Log-loss p-value:{harm_logloss.pvalue:.4f}")
```

On this stratified random split, the p-values should not be especially small. That is the expected
result when training and test come from the same population.

## Interpreting the outcome

- Small p-values mean the test set contains a disproportionate share of higher-error predictions.
- Large p-values mean there is not enough evidence that the model performs worse on the test set.

It is common for Brier score and log-loss to tell a similar story here. `ss.detect_harmful_shift(...)`
is rank-based, so signals that order observations in a similar way often produce similar results.

Use this guide when labels are available. If they are not, use
[Monitor predicted credit risk](monitor-credit-risk.md) or
[Monitor model confidence](monitor-model-confidence.md) instead.
