# How to: Monitor prediction errors once labels arrive

Use this guide when you have ground-truth labels for both groups and want a direct answer about whether the model performs worse.

When labels are available, prediction error is often the cleanest signal you can compare.

## Why this signal works well

Prediction errors turn model quality into a numeric score:

- **Brier score** — squared error on the predicted probability
- **Log-loss** — penalizes confident mistakes more heavily

For both, larger values mean worse predictions, so they work naturally with `ss.test_harmful_shift(..., worse="higher")` (or `ss.Worse.HIGHER`).

## Setup

This guide uses the HELOC dataset with a stratified random split. Unlike the risk/confidence guides (which split by risk level), both groups here come from the same population — so the null is true and p-values should be large.

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
    random_state=12345,
)
```

--8<-- "snippets/honest-scores.txt"

## Step 1 — Fit the model and get honest predictions

Use out-of-bag predictions for training so errors are not artificially optimistic.

```python
rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf.fit(X_train, y_train)

train_prob = rf.oob_decision_function_[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]
```

## Step 2 — Turn predictions into error scores

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

## Step 3 — Test whether errors are worse on the test set

```python
harm_brier = ss.test_harmful_shift(
    source=brier_train,
    target=brier_test,
    worse="higher",
    rng=np.random.default_rng(12345),
)

harm_logloss = ss.test_harmful_shift(
    source=logloss_train,
    target=logloss_test,
    worse="higher",
    rng=np.random.default_rng(12345),
)

print(f"Brier p-value:    {harm_brier.pvalue:.4f}")
print(f"Log-loss p-value: {harm_logloss.pvalue:.4f}")
```

On this stratified random split, expect non-significant p-values (typically > 0.05) — no evidence that the model performs worse on test when both groups come from the same population.

## Interpret the outcome

- **Small p-value (≤ 0.05)** — evidence that test carries a disproportionate share of high-error predictions.
- **Large p-value** — not enough evidence that the model performs worse on test.

Brier and log-loss often tell a similar story here. `ss.test_harmful_shift` is rank-based, so signals that order observations similarly produce similar p-values.

## Next steps

- Without labels, use [Monitor predicted credit risk](monitor-credit-risk.md) or [Monitor model confidence](monitor-model-confidence.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? tip "Why clip probabilities?"
    Log-loss uses `log(p)`. Clipping to `[1e-10, 1-1e-10]` avoids `log(0)` without changing the ranking that the test uses.
