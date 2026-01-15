# Credit (HELOC) Dataset

This example demonstrates testing for **dataset shift** (covariate shift) and **performance degradation** (outcome shift) using the HELOC dataset from [TableShift](https://tableshift.org/datasets.html#heloc). We apply both CTST and DSOS to a credit risk scenario.

## When to use this

- **Dataset shift (CTST)**: Detect when feature distributions differ between training and deployment
- **Performance degradation (DSOS)**: Detect when predictions shift toward worse outcomes

See [Distribution Shifts](distribution-shifts.md) for CTST basics and [Noninferiority](noninferiority.md) for DSOS fundamentals.

## Setup

The HELOC dataset (FICO Explainable AI Challenge) contains credit bureau features. We split it by `ExternalRiskEstimate` to simulate training on one population and deploying to another:

- **Training set** (`ExternalRiskEstimate > 63`): 7,683 low-risk customers
- **Test set** (`ExternalRiskEstimate ≤ 63`): 2,188 high-risk customers

This split creates a domain shift scenario typical in production credit modeling.

## Data Loading

```python
import re
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from samesame.ctst import CTST
from samesame.nit import DSOS

# Fetch HELOC dataset
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

# Find ExternalRiskEstimate column and split data
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_test = X[~mask_high].reset_index(drop=True)
y_test = y[~mask_high].reset_index(drop=True)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

**Output:**

```text
# Training set: 7683 samples (low external risk)
# Test set: 2188 samples (high external risk)
```

## Dataset Shift (CTST)

**Question**: Are the feature distributions different?

**Method**: Train a classifier to distinguish training from test samples. High performance indicates distributional differences. We use OOB predictions for valid inference.

```python
# Create domain labels and concatenate data
split = pd.Series([0] * len(X_train) + [1] * len(X_test))
X_concat = pd.concat([X_train, X_test], ignore_index=True)

# Train RandomForest with OOB predictions
rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
oob_scores = rf_domain.oob_decision_function_[:, 1]

# Run CTST
ctst = CTST(actual=split.values, predicted=oob_scores, metric=roc_auc_score)
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

**Result**: p-value = 0.0002 (highly significant). The classifier easily distinguishes training from test samples, confirming substantial covariate shift between the two populations.

### Feature Importance

To understand *which features* drive the shift, we can examine feature importance:

```python
importances = rf_domain.feature_importances_
feature_names = X_concat.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 5 features distinguishing the datasets:")
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

These importances show which features differ most between the two populations. `ExternalRiskEstimate` dominates because it was the splitting criterion. However, other features like `MSinceMostRecentDelq` and `MaxDelq2PublicRecLast12M` also show some discriminatory power, beyond the splitting variable. Feature importance enables targeted interventions and informs further investigation and analysis.

## Performance Degradation (DSOS)

**Question**: Have predictions shifted toward worse outcomes?

**Method**: Train a credit risk model and compare predicted default probabilities between training and test sets. DSOS tests for adverse shifts in the prediction distribution.

```python
# Train credit risk model
loan_status = y_train.map({'Good': 0, 'Bad': 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, loan_status)

# Get predicted default probabilities
bad_train = rf_bad.oob_decision_function_[:, 1].ravel()
bad_test = rf_bad.predict_proba(X_test)[:, 1].ravel()

# Run DSOS test
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

**Result**: p-value = 0.0001 (highly significant). The model predicts substantially higher default risk for test samples. DSOS detects this adverse shift by focusing on the distribution tail—whether extreme high-risk predictions are more common—without requiring true labels.

## Interpreting Results

**CTST vs. DSOS**: These tests answer complementary questions:

- **CTST**: Do feature distributions differ? (covariate shift)
- **DSOS**: Have predictions shifted adversely? (outcome shift)

**Action guidance**:

- Both significant → Covariate shift causing performance issues. Retrain or recalibrate.
- Only DSOS significant → Label/concept shift without feature changes. Investigate root causes.
- Only CTST significant → Distribution changed but model still generalizes. Monitor closely.
- Neither significant → Safe to operate.

## Practical Recommendations

**1. Monitor Regularly.** Run CTST and DSOS on production batches (weekly/monthly). Automate alerting based on p-value thresholds (e.g., p < 0.01 triggers review).

**2. Investigate Drivers.** Use feature importance to identify which features drive shifts. Distinguish natural drift from data quality issues.

**3. No Labels Needed.** Both tests work without ground truth in deployment, enabling continuous monitoring when labels are delayed.

**4. Respond Appropriately.** Natural drift → monitor. Meaningful shift → recalibrate. Substantial shift → retrain.
