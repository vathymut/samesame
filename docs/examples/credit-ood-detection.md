# How to: Monitor model confidence

**Use this guide when:** your model output is not itself a useful risk signal and you want to
monitor whether deployment predictions look less certain than training predictions.

**What you'll do:**

- Turn model outputs into one confidence value per row
- Compare training and deployment values
- Test whether deployment predictions look less certain

!!! note "Before you start"
    This guide builds directly on the [credit risk how-to](credit-example.md).
    Complete that guide first — it introduces the HELOC dataset and the credit model used here.

This guide uses the same data and model as the [credit risk how-to](credit-example.md), but asks a
different question.

In that guide, the model's **predicted probability of default** was already useful as a
"worse outcome" score. Even without ground-truth labels in production, a higher default probability
is directly interpretable as higher business risk.

That is not always true. In many machine learning tasks, the model's prediction is not itself a
measure of badness. A classifier might predict "cat" versus "dog", or one product category versus
another. Those labels are meaningful predictions, but they do not tell you whether a sample is
**risky**, **unusual**, or **outside the training distribution**.

In those cases, you need a different per-row signal. This guide shows how to use an
**out-of-distribution (OOD) score**, which is a confidence-style value for how unusual an input looks.

## Two kinds of monitoring signals

The key distinction is:

| Signal | What it measures | When to use it |
|--------|------------------|----------------|
| **Predicted default probability** | How likely the model thinks default is | Use when higher predictions already mean worse business outcomes |
| **Confidence value (often called an OOD score)** | How confident the model is in its prediction | Use when the model output is not itself a meaningful risk signal |

In this credit example, default probability and a confidence value are both available, but they answer
different questions:

- **Default probability:** "Does this customer look risky?"
- **Confidence value:** "How confident is the model in this prediction?"

These are related, but not the same. A sample can receive a high-confidence prediction without
necessarily having the highest predicted default probability.

In this guide, LogitGap is used to compare confidence behavior between training and test
predictions, not to directly measure business harm.

That difference matters in practice. A confidence value can move in a reassuring direction because the
model is becoming **more confident**, while the business outcome moves in a harmful direction because
the model is becoming **more confidently wrong** or **more confidently harmful** for the business.
So confidence values should not be read as a direct measure of business safety.

## Why LogitGap?

We use **LogitGap** as the confidence value in this guide.

- **LogitGap** looks at the gap between the model's strongest class score and the remaining class scores.
- A **large** gap means the model is confident in its class decision.
- A **small** gap means the model is uncertain and the sample may be out-of-distribution.

You may also see **MaxLogit** in the literature. It uses only the single largest logit. That is a
reasonable baseline, but LogitGap usually carries more information because it uses the separation
between classes, not just the top score.

For a beginner-friendly workflow, use **LogitGap first** and treat MaxLogit as a simpler baseline worth knowing about.

## Setup

We reuse the same HELOC split as in the [credit risk how-to](credit-example.md):

- **Training set** (`ExternalRiskEstimate > 63`): 7,683 lower-risk customers
- **Deployment set** (`ExternalRiskEstimate ≤ 63`): 2,188 higher-risk customers

```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

from samesame import test_adverse_shift
from samesame.logit_scores import logit_gap, max_logit

# Load the HELOC dataset
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

# Split into training and deployment populations
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_test = X[~mask_high].reset_index(drop=True)

print(f"Training set:   {len(X_train)} samples")
print(f"Deployment set: {len(X_test)} samples")
```

## Step 1 — Train the same credit model

We first train the same credit risk model used in the previous tutorial. The model predicts the
probability of default, but here we are going to reuse its internal outputs to measure
**confidence patterns**, not just risk.

```python
# Train a default-prediction model on the training population
bad_mapping = {'Good': 0, 'Bad': 1}
bad_train = y_train.map(bad_mapping).values

rf_bad = RandomForestClassifier(
        n_estimators=500,
        oob_score=True,
        random_state=12345,
        min_samples_leaf=10,
)
rf_bad.fit(X_train, bad_train)
```

## Step 2 — Convert model outputs into one confidence value per row

`RandomForestClassifier` gives class probabilities. To compute LogitGap, we first convert these
probabilities into **logits**. A logit is the same information written on an open-ended scale
instead of the 0 to 1 probability scale. We then apply `logit_gap`.

We use:

- **OOB predictions** for the training set, so each training point is evaluated by trees that did not train on it
- **Standard predictions** for the deployment set

```python
# Clip probabilities to avoid infinite logits at 0 or 1
train_probs = np.clip(rf_bad.oob_decision_function_, 1e-6, 1 - 1e-6)
test_probs = np.clip(rf_bad.predict_proba(X_test), 1e-6, 1 - 1e-6)

logits_train = logit(train_probs)
logits_test = logit(test_probs)

ood_train = logit_gap(logits_train)
ood_test = logit_gap(logits_test)

print(f"Training mean LogitGap:   {ood_train.mean():.3f}")
print(f"Deployment mean LogitGap: {ood_test.mean():.3f}")

# Optional baseline: MaxLogit (uses only the top logit)
max_train = max_logit(logits_train)
max_test = max_logit(logits_test)
print(f"Training mean MaxLogit:   {max_train.mean():.3f}")
print(f"Deployment mean MaxLogit: {max_test.mean():.3f}")
```

### How to read these values

- **Higher LogitGap**: the model has a larger margin between classes, so it is more confident in its prediction
- **Lower LogitGap**: the model has a smaller margin between classes, so it is less confident in its prediction

This value is primarily about confidence, not direct business harm.
If the deployment distribution shifts downward relative to training, it indicates lower-confidence
predictions in deployment.

For the HELOC split used here, the observed values are:

```text
Training mean LogitGap:   1.8105
Deployment mean LogitGap: 2.1363
Training median LogitGap: 1.5824
Deployment median LogitGap: 2.1617
```

That means the deployment population has **higher**, not lower, LogitGap scores in this example.
So the model appears **more** confident on the deployment population according to this score.

## Step 3 — Plot the value distributions

Before running a formal test, it helps to look at the values directly.

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ood_train, bins=40, alpha=0.6, label="Training", density=True)
ax.hist(ood_test, bins=40, alpha=0.6, label="Deployment", density=True)
ax.set_xlabel("LogitGap score")
ax.set_ylabel("Density")
ax.set_title("Training vs deployment confidence values")
ax.legend()
plt.tight_layout()
plt.show()
```

What to look for:

- If the **deployment** histogram sits noticeably **left** of the training histogram, deployment
    predictions are made with lower confidence.
- If the two histograms largely overlap, there is less evidence of a confidence shift.

For this HELOC split, the deployment histogram should shift **right**, not left, because the
observed LogitGap values are higher in deployment than in training.

## Step 4 — Test the shift with `test_adverse_shift(...)`

Now we turn that confidence shift into a formal hypothesis test.

Higher LogitGap means **higher confidence**, which is better. We express that directly with
`direction="higher-is-better"` instead of negating the values manually.

```python
harm = test_adverse_shift(
    reference=ood_train,
    candidate=ood_test,
    direction="higher-is-better",
)

print("Confidence-value shift test using test_adverse_shift on LogitGap")
print(f"  statistic: {harm.statistic:.4f}")
print(f"  p-value:   {harm.pvalue:.4f}")
```

**Output:**

```text
Confidence-value shift test using test_adverse_shift on LogitGap
    statistic: 0.0409
    p-value:   1.0000
```

### How to interpret the result

- **Small p-value**: strong evidence that deployment contains more low-confidence predictions than training
- **Large p-value**: not enough evidence to claim a confidence drop in deployment

Here the p-value is `1.0000`, so there is **no evidence** that deployment contains more
low-confidence predictions than training. In fact, the LogitGap scores move in the opposite direction:
the deployment customers look *more* confidently classified by this model.

This contrast with the [credit risk how-to](credit-example.md) is the main lesson:

- **Default probability** increased sharply in deployment, so the model predicts worse business outcomes.
- **LogitGap** also increased, so the model does **not** look less confident on deployment data.

Those two findings are not contradictory. They answer different questions. A customer can look
high-risk to the model while still being predicted with high confidence.

This is also the main warning for production monitoring: a confidence-style value can be
misleading if you treat it as a business-risk score. The model can become **more confident** while
its predictions become **more harmful**. Use confidence values to monitor confidence patterns, not to replace a
business outcome metric when such a metric is available.

## When should you use this instead of default probability?

Use **predicted default probability** when the model output already has a clear business meaning,
as it does in the [credit risk how-to](credit-example.md).

Use a **confidence value** when:

- the model output is not itself a risk score
- you want to detect lower-confidence or unusual inputs, not just high-risk predictions
- you need a generic monitoring signal that works across many classification tasks

In practice, the two approaches complement each other:

- **Default probability** tells you whether the model predicts bad outcomes
- **Confidence value** tells you how confidence behavior changes across populations

This HELOC example shows why it is worth monitoring both. Here, default probability detects a clear
adverse shift, while LogitGap does not detect a confidence drop. If you had watched only the confidence
value, you could have missed a harmful business change.

## Summary

This guide uses **LogitGap** as a practical confidence value for beginners.

- It is easy to compute from model outputs
- It is more informative than MaxLogit in most cases
- It works even when the model prediction itself is not an interpretable "worse outcome" score
- Combined with `test_adverse_shift(...)`, it gives you a principled way to test whether deployment confidence degrades relative to training

In this specific credit example, LogitGap does **not** flag deployment as lower confidence. That is a
useful result, not a failure: it shows that confidence and business risk are different concepts and
should be monitored separately. When a business-risk score exists, do not let a confidence value override it.
