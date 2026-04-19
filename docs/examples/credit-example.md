# Tutorial: Monitoring a Credit Risk Model for Shift

**What you'll learn:**

- How to detect whether your production data looks different from your training data (dataset shift)
- How to find which features are responsible for that difference
- How to check whether a model's predictions have shifted toward worse outcomes (performance degradation)
- How to interpret both results and decide what action to take

**Prerequisites:** Familiarity with scikit-learn basics (fitting a model, `predict_proba`).
New to these tests? Read [Detecting distribution shifts](distribution-shifts.md) and
[Noninferiority testing](noninferiority.md) first.

---

## The scenario

You have trained a credit risk model to predict loan default. Your training data came from
**low-risk customers** (good credit history). The model is now deployed and scoring a
**different population** — higher-risk customers.

Two questions arise:

1. **Are the feature distributions different?** If the new customers look nothing like the
   training data, the model may not be reliable on them.
2. **Are the model's predictions worse?** Even if features differ, the model might still
   generalise. What we really care about is whether it is now predicting higher default risk
   — i.e., whether outcomes have shifted adversely.

We will answer both questions using CTST (for question 1) and DSOS (for question 2).
If you want to monitor **model familiarity** instead of **predicted risk**, continue to
[Credit OOD detection](credit-ood-detection.md) after this tutorial.

---

## Setup

We use the **HELOC dataset** (FICO Explainable AI Challenge), which contains credit bureau
features for real customers. We simulate a production deployment scenario by splitting on
`ExternalRiskEstimate`:

- **Training set** (`ExternalRiskEstimate > 63`): 7,683 low-risk customers
- **Deployment set** (`ExternalRiskEstimate ≤ 63`): 2,188 high-risk customers

```python
import re
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from samesame.ctst import CTST
from samesame.nit import DSOS

# Download the HELOC dataset (requires internet access on first run)
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

# Split into training (low-risk) and deployment (high-risk) populations
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_test  = X[~mask_high].reset_index(drop=True)
y_test  = y[~mask_high].reset_index(drop=True)

print(f"Training set:    {len(X_train)} samples")
print(f"Deployment set:  {len(X_test)} samples")
```

**Output:**

```text
Training set:    7683 samples
Deployment set:  2188 samples
```

---

## Step 1 — Detect dataset shift (CTST)

**Question:** Are the feature distributions of the training and deployment sets different?

We train a Random Forest to distinguish training samples from deployment samples.
If the classifier can tell them apart easily (high AUC), the distributions are different.
We use **out-of-bag (OOB) predictions** so that each sample is scored by trees that
never trained on it — this gives us unbiased, valid predictions:

```python
# Label the two populations: 0 = training, 1 = deployment
split = pd.Series([0] * len(X_train) + [1] * len(X_test))
X_concat = pd.concat([X_train, X_test], ignore_index=True)

# Train a classifier to distinguish training from deployment
rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
oob_scores = rf_domain.oob_decision_function_[:, 1]  # probability of being deployment

# Run the CTST
ctst = CTST(actual=split.values, predicted=oob_scores, metric=roc_auc_score)
print(f"AUC statistic: {ctst.statistic:.4f}")
print(f"p-value:       {ctst.pvalue:.4f}")
```

**Output:**

```text
AUC statistic: 1.0000
p-value:       0.0002
```

An AUC of 1.0 means the classifier perfectly separates the two populations.
The p-value of 0.0002 confirms this is far beyond chance — **there is strong evidence of
dataset shift**.

### Which features are driving the shift?

Feature importances from the same classifier tell you which features differ most between
the two populations:

```python
feat_imp = (
    pd.Series(rf_domain.feature_importances_, index=X_concat.columns)
    .sort_values(ascending=False)
)
print("Top 5 features driving the shift:")
print(feat_imp.head(5))
```

**Output:**

```text
Top 5 features driving the shift:
ExternalRiskEstimate          0.642400
MSinceMostRecentDelq          0.069394
MaxDelq2PublicRecLast12M      0.064526
NetFractionRevolvingBurden    0.050656
PercentTradesNeverDelq        0.042478
```

`ExternalRiskEstimate` dominates because it was used to create the split — that is expected.
Interestingly, several other features (`MSinceMostRecentDelq`, `MaxDelq2PublicRecLast12M`) also
differ between the groups, which suggests that the features may be correlated.

---

## Step 2 — Test for performance degradation (DSOS)

**Question:** Has the model started predicting worse outcomes for the deployment population?

Even though the feature distributions are different, the model might still generalise.
We now check whether the model's predicted default probabilities are higher (worse) for
deployment samples than for training samples.

We train a credit risk model on the training set and compare its predictions on both populations.
OOB predictions are used for the training set to avoid inflated scores:

```python
# Train a credit risk model to predict loan default (Bad = 1)
loan_status = y_train.map({'Good': 0, 'Bad': 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, loan_status)

# Predicted default probability for each group
# Training: OOB predictions (unbiased); Deployment: standard predictions
bad_train = rf_bad.oob_decision_function_[:, 1].ravel()
bad_test  = rf_bad.predict_proba(X_test)[:, 1].ravel()

# Run DSOS: are there disproportionately more high-risk predictions in deployment?
dsos = DSOS.from_samples(bad_train, bad_test)
print(f"WAUC statistic: {dsos.statistic:.4f}")
print(f"p-value:        {dsos.pvalue:.4f}")
```

**Output:**

```text
WAUC statistic: 0.2483
p-value:        0.0001
```

p = 0.0001 — **strong evidence of adverse shift**. The model is predicting substantially
higher default risk for deployment samples. This confirms not only that the data is different,
but that the difference is harmful: predictions have shifted toward worse outcomes.

This is a good example of when the model output itself is already meaningful. A higher predicted
default probability is directly interpretable as higher business risk, so it is a natural score to
monitor. When a model output is *not* directly interpretable as "worse", you need a different score,
such as an out-of-distribution score. See [Credit OOD detection](credit-ood-detection.md).

The important limitation is the reverse: an OOD score is **not** a substitute for business impact.
A model can become more confident in its predictions while those predictions become more harmful to
the business. When you already have a score with direct business meaning, such as default probability,
that score should remain the primary monitoring signal.

---

## Step 3 — Interpret the combined results

Running both CTST and DSOS together gives a richer picture than either test alone:

| Scenario                         | Recommended action                                   |
|----------------------------------|------------------------------------------------------|
| Both CTST and DSOS significant   | Data and outcomes have shifted. Retrain or recalibrate the model. |
| Only CTST significant            | Data looks different, but outcomes haven't shifted. Monitor closely. |
| Only DSOS significant            | Outcome shift without feature change (concept drift). Investigate root causes. |
| Neither significant              | No evidence of a problem. Continue as normal.        |

In this example, **both tests are significant** — the deployment population is different
and the model's predictions are worse. The recommended action is to retrain or recalibrate
the model for the new population.

---

## Key takeaways

- **CTST** detects whether feature distributions differ between training and deployment.
  Feature importances help identify *which* features are responsible.
- **DSOS** detects whether the model's predictions have shifted adversely. It does not require
  ground truth labels, making it practical for production monitoring before labels arrive.
- Use **both tests together** for a complete picture: CTST tells you *what* changed,
  DSOS tells you *whether it matters*.
- In this example, **predicted risk increased**, but in the companion OOD tutorial, **model
  familiarity did not worsen**. Those are different signals and both are worth monitoring.
- If your model output is not itself a meaningful risk score, use an OOD score instead; see
  [Credit OOD detection](credit-ood-detection.md).
