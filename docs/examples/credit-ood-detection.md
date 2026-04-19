# Tutorial: Detecting Out-of-Distribution Customers with LogitGap

This tutorial is a continuation of the [Credit Example](credit-example.md), but it answers a
different question.

In the credit example, the model's **predicted probability of default** was already useful as a
"worse outcome" score. Even without ground-truth labels in production, a higher default probability
is directly interpretable as higher business risk.

That is not always true. In many machine learning tasks, the model's prediction is not itself a
measure of badness. A classifier might predict "cat" versus "dog", or one product category versus
another. Those labels are meaningful predictions, but they do not tell you whether a sample is
**risky**, **unusual**, or **outside the training distribution**.

In those cases, you need a different score. This tutorial shows how to use an
**out-of-distribution (OOD) score** as a proxy for worse outcomes.

## What you will learn

- How OOD scores differ from predicted default probabilities
- Why **LogitGap** is a better default choice than **MaxLogit**
- How to visualize OOD scores for training and deployment data
- How to use DSOS to test whether deployment samples look less familiar to the model

## Two kinds of scores

The key distinction is:

| Score type | What it measures | When to use it |
|------------|------------------|----------------|
| **Predicted default probability** | How likely the model thinks default is | Use when higher predictions already mean worse business outcomes |
| **OOD score** | How familiar the sample looks to the model | Use when the model output is not itself a meaningful risk score |

In this credit example, default probability and OOD score are both available, but they answer
different questions:

- **Default probability:** "Does this customer look risky?"
- **OOD score:** "Does this customer look familiar to the model?"

These are related, but not the same. A sample can be unfamiliar without necessarily having the
highest predicted default probability.

That difference matters in practice. An OOD score can move in a reassuring direction because the
model is becoming **more confident**, while the business outcome moves in a harmful direction because
the model is becoming **more confidently wrong** or **more confidently harmful** for the business.
So OOD scores should not be read as a direct measure of business safety.

## Why LogitGap?

We use **LogitGap** as the OOD score in this tutorial.

- **LogitGap** looks at the gap between the model's strongest class score and the remaining class scores.
- A **large** gap means the model is confident and the sample looks familiar.
- A **small** gap means the model is uncertain and the sample may be out-of-distribution.

You may also see **MaxLogit** in the literature. It uses only the single largest logit. That is a
reasonable baseline, but LogitGap usually carries more information because it uses the separation
between classes, not just the top score.

For a novice workflow: use **LogitGap first** and treat MaxLogit as a simpler baseline worth knowing about.

## Setup

We reuse the same HELOC split as in the [Credit Example](credit-example.md):

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

from samesame.nit import DSOS
from samesame.ood import logit_gap

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
probability of default, but here we are going to reuse its internal scores to measure
**familiarity**, not just risk.

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

## Step 2 — Convert model outputs into OOD scores

`RandomForestClassifier` gives class probabilities. To compute LogitGap, we first convert these
probabilities into logits. We then apply `logit_gap`.

We use:

- **OOB predictions** for the training set, so each training point is scored by trees that did not train on it
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
```

### How to read these scores

- **Higher LogitGap**: the model is more confident, so the sample looks more like the training data
- **Lower LogitGap**: the model is less confident, so the sample looks less familiar

If the deployment distribution shifts downward relative to the training distribution, that is a sign
the model is seeing more unusual customers.

For the HELOC split used here, the observed values are:

```text
Training mean LogitGap:   1.8105
Deployment mean LogitGap: 2.1363
Training median LogitGap: 1.5824
Deployment median LogitGap: 2.1617
```

That means the deployment population has **higher**, not lower, LogitGap scores in this example.
So the model appears **more** confident on the deployment population according to this score.

## Step 3 — Plot the score distributions

Before running a formal test, it helps to look at the scores directly.

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ood_train, bins=40, alpha=0.6, label="Training", density=True)
ax.hist(ood_test, bins=40, alpha=0.6, label="Deployment", density=True)
ax.set_xlabel("LogitGap score")
ax.set_ylabel("Density")
ax.set_title("Training vs deployment OOD scores")
ax.legend()
plt.tight_layout()
plt.show()
```

What to look for:

- If the **deployment** histogram sits noticeably **left** of the training histogram, deployment
    samples look less familiar to the model.
- If the two histograms largely overlap, there is less evidence of an OOD shift.

For this HELOC split, the deployment histogram should shift **right**, not left, because the
observed LogitGap values are higher in deployment than in training.

## Step 4 — Test the shift with DSOS

Now we turn the score shift into a formal hypothesis test.

DSOS expects **higher** scores to mean "worse". But higher LogitGap means **more familiar**, which
is the opposite of what we want. So we negate the scores before passing them into DSOS.

```python
dsos_ood = DSOS.from_samples(-ood_train, -ood_test)

print("OOD shift test using DSOS on LogitGap")
print(f"  statistic: {dsos_ood.statistic:.4f}")
print(f"  p-value:   {dsos_ood.pvalue:.4f}")
```

**Output:**

```text
OOD shift test using DSOS on LogitGap
    statistic: 0.0409
    p-value:   1.0000
```

### How to interpret the result

- **Small p-value**: strong evidence that deployment contains more low-familiarity samples than training
- **Large p-value**: not enough evidence to claim an OOD shift

Here the p-value is `1.0000`, so there is **no evidence** that deployment contains more
low-familiarity samples than training. In fact, the LogitGap scores move in the opposite direction:
the deployment customers look *more* confidently classified by this model.

This contrast with the [Credit Example](credit-example.md) is the main lesson:

- **Default probability** increased sharply in deployment, so the model predicts worse business outcomes.
- **LogitGap** also increased, so the model does **not** look less confident on deployment data.

Those two findings are not contradictory. They answer different questions. A customer can look
high-risk to the model while still looking familiar to the model.

This is also the main warning for production monitoring: a confidence-style OOD score can be
misleading if you treat it as a business-risk score. The model can become **more confident** while
its predictions become **more harmful**. Use OOD scores to monitor familiarity, not to replace a
business outcome metric when such a metric is available.

## When should you use this instead of default probability?

Use **predicted default probability** when the model output already has a clear business meaning,
as it does in the [Credit Example](credit-example.md).

Use an **OOD score** when:

- the model output is not itself a risk score
- you want to detect unfamiliar inputs, not just high-risk predictions
- you need a generic monitoring signal that works across many classification tasks

In practice, the two approaches complement each other:

- **Default probability** tells you whether the model predicts bad outcomes
- **OOD score** tells you whether the model is operating on unfamiliar data

This HELOC example shows why it is worth monitoring both. Here, default probability detects a clear
adverse shift, while LogitGap does not detect a familiarity problem. If you had watched only the OOD
score, you could have missed a harmful business change.

## Key takeaway

This tutorial uses **LogitGap** as a practical OOD score for novice users.

- It is easy to compute from model outputs
- It is more informative than MaxLogit in most cases
- It works even when the model prediction itself is not an interpretable "worse outcome" score
- Combined with DSOS, it gives you a principled way to test whether deployment data looks less familiar than training data

In this specific credit example, LogitGap does **not** flag deployment as less familiar. That is a
useful result, not a failure: it shows that familiarity and business risk are different concepts and
should be monitored separately. When a business-risk score exists, do not let an OOD score override it.
