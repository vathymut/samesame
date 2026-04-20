# How to: Monitor prediction errors with per-sample scores

**Use this guide when:** ground-truth labels are available for both the training and test sets,
and you want to test whether the model’s per-sample prediction errors are systematically higher
on the test set.

**What you’ll do:**

- Fit a credit risk model on a random training split using out-of-bag predictions
- Compute per-sample Brier scores and log-losses
- Test whether either score is adversely shifted between training and test

!!! note "Before you start"
    This guide assumes you have completed both tutorials:

  - [Detect a distribution shift](/examples/tutorials/detect-distribution-shift.md)
  - [Check whether a shift is harmful](/examples/tutorials/check-shift-harm.md)

    You also need basic familiarity with scikit-learn — fitting a model and calling \`predict_proba\`.

---

## The scenario

When ground-truth labels are available for a test set, per-sample prediction errors provide a
direct measure of model accuracy on each row. Two standard choices are the **Brier score** and
**log-loss**.

For a predicted probability $\hat{p}$ and true label $y \in \{0, 1\}$:

- **Brier score:** $(y - \hat{p})^2$ — the squared difference between the true label and the
  predicted probability.
- **Log-loss:** $-[y \log \hat{p} + (1-y)\log(1-\hat{p})]$ — penalises overconfident wrong
  predictions more heavily than the Brier score.

For both scores, larger values mean worse predictions. They can therefore serve directly as the
per-sample adversity score that \`test_adverse_shift(...)\` expects.

Note that these scores require labels. They are not available during production monitoring when
outcomes are delayed, but they are appropriate for evaluating a held-out test set or a labelled
historical batch.

This guide complements the other monitoring guides:

- Use [Monitor a credit risk model](/examples/credit/monitor-credit-risk.md) when you need a label-free business-risk signal.
- Use [Monitor model confidence](/examples/credit/monitor-confidence-ood.md) when you need a label-free confidence signal.
- Use this guide when labels are available and you want direct per-sample error measures.

---

## Setup

We use the **HELOC dataset** (FICO Explainable AI Challenge), split randomly into training and
test sets. Unlike the [credit risk how-to](/examples/credit/monitor-credit-risk.md), this split is not based on a
feature threshold — it is a stratified random split, so both sets are drawn from the same
population.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from samesame import test_adverse_shift

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

y_binary = (y == "Bad").astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.30,
    stratify=y_binary,
    random_state=12345,
)

print(f"Training set: {len(X_train)} samples,  default rate: {y_train.mean():.4f}")
print(f"Test set:     {len(X_test)} samples,  default rate: {y_test.mean():.4f}")
```

**Output:**

```text
Training set: 6909 samples,  default rate: 0.5203
Test set:     2962 samples,  default rate: 0.5203
```

The default rate is equal in both sets because `stratify=y_binary` preserves it.

---

## Step 1 — Fit the model

Fit a Random Forest with `oob_score=True`. Out-of-bag (OOB) predictions will be used for the
training set to avoid evaluating the model on data it was trained on — doing so would produce
artificially low error scores and bias the comparison.

```python
rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf.fit(X_train, y_train)

p_train = rf.oob_decision_function_[:, 1]  # OOB predictions for training set
p_test  = rf.predict_proba(X_test)[:, 1]   # standard predictions for test set
```

---

## Step 2 — Compute per-sample prediction errors

Each row receives one error score. Both Brier score and log-loss are computed from the true label
and the predicted probability for that row.

```python
# Brier score: squared difference between predicted probability and true label
brier_train = (y_train - p_train) ** 2
brier_test  = (y_test  - p_test)  ** 2

# Log-loss: log-probability assigned to the correct label (clipped to avoid log(0))
eps = 1e-10
p_tr = np.clip(p_train, eps, 1 - eps)
p_te = np.clip(p_test,  eps, 1 - eps)
logloss_train = -(y_train * np.log(p_tr) + (1 - y_train) * np.log(1 - p_tr))
logloss_test  = -(y_test  * np.log(p_te) + (1 - y_test)  * np.log(1 - p_te))

print(f"Mean Brier score — training: {brier_train.mean():.4f},  test: {brier_test.mean():.4f}")
print(f"Mean log-loss    — training: {logloss_train.mean():.4f},  test: {logloss_test.mean():.4f}")
```

**Output:**

```text
Mean Brier score — training: 0.1806,  test: 0.1830
Mean log-loss    — training: 0.5412,  test: 0.5463
```

Both scores are slightly higher on the test set, but the means are close. The question is whether
this difference is consistent with random variation or reflects a systematic pattern.

---

## Step 3 — Test for adverse shift

Both scores are "higher is worse", so we pass `direction="higher-is-worse"`.

```python
harm_brier = test_adverse_shift(
    reference=brier_train,
    candidate=brier_test,
    direction="higher-is-worse",
)

harm_logloss = test_adverse_shift(
    reference=logloss_train,
    candidate=logloss_test,
    direction="higher-is-worse",
)

print(f"Brier score — statistic: {harm_brier.statistic:.4f},  p-value: {harm_brier.pvalue:.4f}")
print(f"Log-loss    — statistic: {harm_logloss.statistic:.4f},  p-value: {harm_logloss.pvalue:.4f}")
```

**Output:**

```text
Brier score — statistic: 0.0846,  p-value: 0.2728
Log-loss    — statistic: 0.0846,  p-value: 0.2744
```

---

## Reading the results

| p-value        | What it means |
|----------------|---------------|
| Small (< 0.05) | Evidence that the test set contains a disproportionate share of high-error predictions |
| Large (≥ 0.05) | Not enough evidence to conclude the model performs worse on the test set |

Here, p ≈ 0.27 for both scores. This is expected: both sets were drawn from the same population,
so there is no reason to expect the model to perform systematically worse on the test set.

Contrast this with the [credit risk how-to](/examples/credit/monitor-credit-risk.md), where a deliberate population
split produces a highly significant result (p = 0.0001). In that guide, the test set contains
structurally different, higher-risk customers. Here, stratified random splitting ensures the two
sets are comparable, and the test correctly finds no evidence of adverse shift.

---

## Why both scores give the same test statistic

`test_adverse_shift` uses a **rank-based statistic**: it compares how the two samples rank
together, not their raw values. For a given label $y$, both Brier score and log-loss are monotone
functions of the predicted probability $\hat{p}$, so their rankings across rows are identical.
The test statistic is therefore the same.

The choice between the two scores is a matter of interpretation:

- **Brier score** is bounded between 0 and 1 and penalises all errors quadratically.
- **Log-loss** is unbounded and penalises overconfident wrong predictions more heavily.

From a testing standpoint, either score is sufficient for binary labels. Report both if you want
to communicate the result to audiences familiar with different conventions.

---

## When to use each monitoring signal

| Signal | Labels required? | Best used when |
|--------|------------------|----------------|
| Predicted default probability | No | Labels are unavailable; the model output has direct business meaning |
| Brier score / log-loss | Yes | A labelled test set is available; you want a direct measure of prediction accuracy |
| LogitGap (confidence score) | No | The model output is not a meaningful risk score; you want to monitor prediction confidence |

For production monitoring before labels arrive, use predicted probability or a confidence score.
When labels become available, Brier score or log-loss provides a direct measurement and can
confirm or revise the earlier assessment.

---

## Summary

- **Per-sample prediction errors** require ground-truth labels but directly measure how wrong the
  model was on each row.
- Use **OOB predictions** for the training set to avoid evaluating the model on data it was
  trained on, which would produce artificially low error scores.
- For this stratified random split, neither score shows significant adverse shift (p ≈ 0.27) —
  the expected result when both sets are drawn from the same population.
- For an example where adverse shift is detected, see [Monitor a credit risk model](/examples/credit/monitor-credit-risk.md).
- For label-free monitoring, see [Monitor a credit risk model](/examples/credit/monitor-credit-risk.md) (predicted
  probability) or [Monitor model confidence](/examples/credit/monitor-confidence-ood.md) (confidence scores).
