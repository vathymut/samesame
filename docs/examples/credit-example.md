# Credit (HELOC) Dataset

This example demonstrates how `samesame` can be used on a real-world credit dataset to detect both **dataset shift** (when the data distribution changes between training and deployment) and **performance degradation** (when models make worse predictions). We use the HELOC (Home Equity Line of Credit) dataset from [TableShift](https://tableshift.org/), as described in the [HELOC dataset documentation](https://tableshift.org/datasets.html#heloc).

## When to use this

- **Dataset shift**: Your training data comes from one population but your model is deployed on a different population. You need to detect when this happens.
- **Performance monitoring**: You need to know if the model predictions (e.g., default probabilities) have shifted in a harmful (adverse) way over time.

This example shows both approaches applied to credit risk scoring.

## Context

The HELOC dataset comes from the FICO Community Explainable AI Challenge, an open-source dataset containing features derived from anonymized credit bureau data. It includes credit history and risk assessment data. The dataset naturally splits into two populations based on `ExternalRiskEstimate`:

1. **Low credit-risk segment** (`External Risk Estimate > 63`): Lower risk customers.
2. **High credit-risk segment** (`External Risk Estimate ≤ 63`): Higher risk customers.

We'll treat the low-risk segment as training data and the high-risk segment as a held-out deployment domain, simulating a real-world scenario where a model trained on one population is deployed to a different population with different risk profiles.

## Dataset Preparation

```python
import re
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from samesame.ctst import CTST
from samesame.nit import DSOS

# Fetch the HELOC dataset from OpenML
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

# Identify the External Risk Estimate column (case-insensitive lookup)
# This column is used to split the data into two distinct populations
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_candidates = (c for c in X.columns if re_obj.search(c))
col_split = next(col_candidates, None)
if col_split is None:
    raise ValueError("External Risk Estimate column not found")

# Split data by external risk threshold (63 is the standard cutoff)
threshold = 63
mask_high = X[col_split].astype(float) > threshold

# Training domain: low credit-risk customers (high external risk estimates)
X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)

# Deployment domain: high credit-risk customers (low external risk estimates)
# This represents the domain shift scenario
X_test = X[~mask_high].reset_index(drop=True)
y_test = y[~mask_high].reset_index(drop=True)

print(f"Training domain: {len(X_train)} samples (low external risk)")
print(f"Held-out domain: {len(X_test)} samples (high external risk)")
```

**Output:**

```text
# Training domain: 7683 samples (low external risk)
# Held-out domain: 2188 samples (high external risk)
```

## Dataset Shift

**Question**: Are the two domains statistically different?

**Method**: We train a classifier to distinguish between the two domains. A high-performing classifier indicates meaningful differences. We use out-of-bag (OOB) predictions to avoid data leakage and obtain valid p-values in keeping with the Classifier Two-Sample Test (CTST) framework.

```python
# Create binary domain labels: 0 = training, 1 = held-out (deployment)
membership = pd.Series([0] * len(X_train) + [1] * len(X_test))
X_concat = pd.concat([X_train, X_test], ignore_index=True)

# Train a RandomForest classifier to distinguish between domains
# Key settings:
#   - oob_score=True: Enables OOB predictions for unbiased model evaluation
#   - n_estimators=500: Number of trees (higher = more robust estimates)
#   - min_samples_leaf=10: Minimum samples per leaf (prevents overfitting)
rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, membership)

# Extract OOB predicted probabilities (probability of being in held-out domain)
oob_scores = rf_domain.oob_decision_function_[:, 1]

# Run Classifier Two-Sample Test (CTST) using AUC as the test metric
# CTST tests whether the classifier's discrimination ability is statistically significant
ctst = CTST(actual=membership.values, predicted=oob_scores, metric=roc_auc_score)
print(f"Dataset Shift Detection (CTST)")
print(f"  Statistic: {ctst.statistic:.4f}")
print(f"  p-value: {ctst.pvalue:.4f}")
```

**Output:**

```text
# Dataset Shift Detection (CTST)
#   Statistic: 1.0000
#   p-value: 0.0002
```

**Interpretation**: The p-value = 0.0002 is highly significant (p < 0.05), indicating the training and held-out domains are statistically different. This is expected since we deliberately split the data by risk profile. A classifier that can reliably distinguish between these domains reveals that their underlying feature distributions differ substantially.

### Feature Importance

To understand *which features* drive the domain shift, we can examine feature importance:

```python
importances = rf_domain.feature_importances_
feature_names = X_concat.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 5 features distinguishing the domains:")
print(feat_imp.head(5))
```

**Output:**

```text
Top 5 features distinguishing the domains:
ExternalRiskEstimate          0.642400
MSinceMostRecentDelq          0.069394
MaxDelq2PublicRecLast12M      0.064526
NetFractionRevolvingBurden    0.050656
PercentTradesNeverDelq        0.042478
dtype: float64
```

These features reveal which characteristics differ most between the two populations. `ExternalRiskEstimate` dominates because it was the splitting criterion. However, other features like `MSinceMostRecentDelq` and `MaxDelq2PublicRecLast12M` also show notable importance, indicating the populations differ beyond just the splitting variable. This suggests **covariate shift**: the feature distributions differ, but the label distribution may respond differently.

## Performance Degradation

**Question**: Has the model's prediction behavior changed in a way that suggests degradation?

**Method**: We train a credit risk model on the training domain to predict loan default (`y`). Then we compare the model's predicted default probabilities between the training and deployment domains. The Dataset-Shift-with-Outlier-Scores (DSOS) test detects whether the tail of the distribution has shifted, indicating more worse outcomes.

```python
# Train a credit risk model on the training domain to predict loan default
# Convert labels from categorical (Good/Bad) to numeric (0/1)
loan_status = y_train.map({'Good': 0, 'Bad': 1}).values

# Train RandomForest to predict probability of default
rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, loan_status)

# Extract predicted default probabilities
# - Training domain: Use OOB predictions (unbiased, non-resubstitution)
# - Deployment domain: Use standard predictions on held-out data
bad_train = rf_bad.oob_decision_function_[:, 1].ravel()
bad_test = rf_bad.predict_proba(X_test)[:, 1].ravel()

# Apply Degree-of-Shift Outcome-Shift (DSOS) test
# Tests whether the distribution of predictions has shifted toward worse outcomes
dsos = DSOS.from_samples(bad_train, bad_test)
print(f"\nPerformance Degradation (DSOS)")
print(f"  Statistic: {dsos.statistic:.4f}")
print(f"  p-value: {dsos.pvalue:.4f}")
```

**Output:**

```text
# Performance Degradation (DSOS)
#   Statistic: 0.2483
#   p-value: 0.0001
```

**Interpretation**: The p-value = 0.0001 is highly significant, indicating the model's predicted default probabilities have shifted substantially upward in the deployment domain. This reveals **performance degradation**: the model now predicts higher risk for the new population. Unlike a t-test (which measures mean shift), DSOS focuses on whether the *tail* of the distribution has shifted, capturing whether more extreme high-risk predictions are now common—the key indicator of adverse performance.

## Interpreting the Results

In this example, we observe:

- **CTST (p = 0.0002, rejected)**: The held-out domain is significantly different from training. The feature distributions have shifted due to our intentional split by risk profile.
- **DSOS (p = 0.0001, rejected)**: The model's predicted default probabilities have shifted upward. This indicates **performance degradation**—the model predicts substantially higher risk for the new population, suggesting real concerns about model reliability in production.
- **Feature importance**: `ExternalRiskEstimate` dominates, but other features also differ between populations, indicating both direct and indirect covariate shifts.

## Key Insights

1. **Dataset Shift vs. Performance Degradation**:
   - **CTST** tests if the *feature distributions* differ between domains. A significant result means covariate shift has occurred.
   - **DSOS** tests if the *model's predictions* have shifted toward worse outcomes. A significant result means the model's behavior has degraded.
   - Together, they tell complementary stories: CTST detects the problem exists, DSOS confirms it affects your model.

2. **Four Possible Scenarios**:
   - CTST non-sig, DSOS non-sig: Domains are similar, model performs consistently. ✓ Safe
   - CTST sig, DSOS non-sig: Domains differ but model generalizes well. ✓ Monitor
   - CTST sig, DSOS sig: Domain shift degrades model performance. ✗ Retrain
   - CTST non-sig, DSOS sig: Subtle shift affecting predictions despite similar features. ✗ Investigate

3. **Feature-Level Understanding**: Feature importance reveals *which* features drive the shift, enabling targeted interventions (e.g., recalibration, retraining, or domain adaptation on key features).

## Practical Recommendations

1. **Implement Monitoring**: Regularly compute CTST and DSOS on production data batches to detect domain shift and performance degradation early.

2. **Set Action Thresholds**: Define decision rules and automate responses:
   - **p < 0.01**: Trigger alerts and schedule urgent review
   - **p < 0.05**: Log flag and prepare retraining pipeline
   - **p > 0.10**: Continue normal operation

3. **Investigate Root Causes**: When tests are significant:
   - Compute feature importance to identify which features shifted most
   - Analyze data summaries (distributions, percentiles) for each top feature
   - Determine if shift is natural drift or a data quality issue

4. **Choose Appropriate Metrics**: You don't need true labels to monitor performance. Good proxies include:
   - Predicted probabilities (as in this example)
   - Model confidence scores
   - Feature distributions
   - Derived metrics (e.g., average predicted default rate)

5. **Plan Remediation**: Based on shift severity and business impact, decide whether to:
   - Monitor closely but take no action (domains naturally drift)
   - Recalibrate the existing model
   - Retrain on more recent data
   - Implement domain adaptation techniques
