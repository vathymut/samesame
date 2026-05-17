# How to: Monitor model confidence

**Use this guide when:** your model output is not itself a directly interpretable risk signal and you want to
monitor whether deployment predictions appear less certain than training predictions.

**What you'll do:**

- Turn model outputs into a Logit-derived Outlier score
- Compare training and deployment Outlier scores
- Test whether deployment predictions look less certain

!!! note "Companion guide"
    This guide uses the same HELOC scenario as [Monitor a credit risk model](monitor-credit-risk.md).
    Read that guide if you want the business-risk comparison. This guide can be run on its own.

This guide uses the same data and model as the [credit risk how-to](monitor-credit-risk.md),
but focuses on confidence rather than business risk. In the credit-risk guide, predicted default
probability has direct business meaning. Here, we use a **Logit-derived Outlier score** as a
confidence-monitoring signal.

If ground-truth labels are available for the deployment set, prediction errors for each row
(Brier score, log-loss) provide a more direct measure of model accuracy; see
[Monitor prediction errors when labels are available](monitor-prediction-errors.md).

## Two kinds of monitoring signals

The key distinction is:

| Signal | What it measures | When to use it |
|--------|------------------|----------------|
| **Predicted default probability** | How likely the model thinks default is | Use when higher predictions already mean worse business outcomes |
| **Logit-derived Outlier score** | How separated the model's class evidence is | Use when the model output is not itself a meaningful risk signal |

In this credit example, default probability and an Outlier score are both available, but they answer
different questions:

- **Default probability:** "Does this customer look risky?"
- **Logit-derived Outlier score:** "How confident is the model in this prediction?"

These are related, but not the same. A sample can receive a high-confidence prediction without
necessarily having the highest predicted default probability.

In this guide, `LogitGap` is used to compare confidence behavior between training and deployment
predictions, not to directly measure business harm.

That difference matters in practice. An Outlier score can move in a reassuring direction because the
model is becoming **more confident**, while the business outcome moves in a harmful direction because
the model is becoming **more confidently wrong** or **more confidently harmful** for the business.
Use this Outlier score to monitor confidence patterns, not to replace a business outcome metric when
such a metric is available.

## Why LogitGap?

We use **LogitGap** as the primary **Logit-derived Outlier score** in this guide.

- **LogitGap** looks at the gap between the model's strongest class logit and the remaining class logits.
- A **large** gap means the model is more confident in its class decision.
- A **small** gap means the model is less confident.

You may see **MaxLogit** in the literature as a simpler alternative, but this guide keeps `LogitGap`
as the only runnable recipe because it uses the separation between classes, not just the top score.

## Setup

This guide uses the same HELOC dataset and split as the [credit risk how-to](monitor-credit-risk.md).

```python
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
import samesame as ss

# Load the HELOC dataset
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

# Split into training and deployment populations
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)

print(f"Training set:   {len(X_train)} samples")
print(f"Deployment set: {len(X_deployment)} samples")
```

## Step 1 — Train the credit model

Reuse the same credit model as the [credit risk how-to](monitor-credit-risk.md). Here we extract its
outputs to measure **confidence patterns**, not risk.

```python
bad_mapping = {"Good": 0, "Bad": 1}
bad_train = y_train.map(bad_mapping).values

rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, bad_train)
```

## Step 2 — Build a Logit-derived Outlier score

`RandomForestClassifier` does not expose native logits. In this example, we build a
**Logit-derived Outlier score** from class probabilities using a guide-owned helper Module.
This is the exact code path exercised by the regression test for this guide.

We use:

- **OOB predictions** for the training set, so each training point is evaluated by trees that did not train on it
- **Standard predictions** for the deployment set

```python
--8<-- "_code/monitor_model_confidence_example.py:imports"
--8<-- "_code/monitor_model_confidence_example.py:logit-gap"
--8<-- "_code/monitor_model_confidence_example.py:outlier-scores"

train_probs = rf_bad.oob_decision_function_
deployment_probs = rf_bad.predict_proba(X_deployment)

train_outlier_scores = outlier_scores_from_probabilities(train_probs)
deployment_outlier_scores = outlier_scores_from_probabilities(deployment_probs)

print(f"Training mean LogitGap:   {train_outlier_scores.mean():.3f}")
print(f"Deployment mean LogitGap: {deployment_outlier_scores.mean():.3f}")
```

On this HELOC split, deployment Outlier scores are higher on average than training Outlier scores.
That means the model appears **more** confident on the deployment population according to this
Outlier score.

### How to read these Outlier scores

- **Higher LogitGap**: the model has a larger margin between classes, so it is more confident
- **Lower LogitGap**: the model has a smaller margin between classes, so it is less confident

This Outlier score is primarily about confidence, not direct business harm. If the deployment
distribution shifts downward relative to training, it indicates lower-confidence predictions in
deployment.

## Step 3 — Plot the Outlier score distributions

Before running a formal test, it helps to inspect the Outlier score distributions directly.

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(train_outlier_scores, bins=40, alpha=0.6, label="Training", density=True)
ax.hist(deployment_outlier_scores, bins=40, alpha=0.6, label="Deployment", density=True)
ax.set_xlabel("LogitGap Outlier score")
ax.set_ylabel("Density")
ax.set_title("Training vs deployment Outlier scores")
ax.legend()
plt.tight_layout()
plt.show()
```

Interpretation:

- If the **deployment** histogram sits noticeably **left** of the training histogram, deployment
  predictions are made with lower confidence.
- If the two histograms largely overlap, there is less evidence of a confidence shift.

For this HELOC split, the deployment histogram should shift **right**, not left, because the
deployment Outlier scores are higher than the training Outlier scores.

## Step 4 — Test the confidence workflow with `ss.shift.detect_harm(...)`

Higher LogitGap means **higher confidence**, which is better. We express that directly with
`direction="higher-is-better"` instead of negating the values manually.

This is the handoff from the guide's training and deployment language to the package's
Source and Target seam.

```python
source_scores = train_outlier_scores
target_scores = deployment_outlier_scores

harm = ss.shift.detect_harm(
    source=source_scores,
    target=target_scores,
    direction="higher-is-better",
    random_state=12345,
)

print("Confidence Outlier-score harmful-shift test on LogitGap")
print(f"  statistic: {harm.statistic:.4f}")
print(f"  p-value:   {harm.pvalue:.4f}")
```

### How to interpret the result

| p-value         | What it means |
|-----------------|---------------|
| Small (< 0.05)  | Evidence that deployment contains a disproportionate share of low-confidence predictions |
| Large (≥ 0.05)  | Not enough evidence to claim a confidence drop in deployment |

On this HELOC split, this workflow should **not** indicate a harmful confidence shift. The
deployment Outlier scores move in the opposite direction: deployment customers look more
confidently classified by this model.

This contrast with the [credit risk how-to](monitor-credit-risk.md) is the main lesson:

- **Default probability** increased sharply in deployment, so the model predicts worse business outcomes.
- **LogitGap** also increased, so the confidence-monitoring workflow does **not** indicate lower confidence in deployment.

Those two findings are not contradictory. They answer different questions. A customer can look
high-risk to the model while still being predicted with high confidence.

## When should you use this instead of default probability?

Use **predicted default probability** when the model output already has a clear business meaning,
as it does in the [credit risk how-to](monitor-credit-risk.md).

Use a **Logit-derived Outlier score** when:

- the model output is not itself a risk score
- you want to detect lower-confidence or unusual inputs, not just high-risk predictions
- you need a generic confidence-monitoring signal that works across many classification tasks

In practice, the two approaches complement each other:

- **Default probability** tells you whether the model predicts bad outcomes
- **Outlier score** tells you how confidence behavior changes across populations

This HELOC example shows why it is worth monitoring both. Here, default probability detects a clear
adverse shift, while LogitGap does not indicate a confidence drop. If you had watched only the
confidence workflow, you could have missed a harmful business change.

## Summary

This guide uses **LogitGap** as a practical **Logit-derived Outlier score** for confidence monitoring.

- It is easy to compute from class probabilities after a logit transform.
- It gives a reusable confidence-monitoring signal when the model output is not itself a meaningful business-risk score.
- Combined with `ss.shift.detect_harm(...)`, it lets you test whether deployment confidence degrades relative to training.

For a direct business-risk signal, use [Monitor a credit risk model](monitor-credit-risk.md).
For label-based monitoring, use [Monitor prediction errors](monitor-prediction-errors.md).