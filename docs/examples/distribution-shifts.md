# Test for Distribution Shifts

## Overview

Given two datasets `sample_P` and `sample_Q` from distributions $P$ and $Q$, the goal is to estimate a $p$-value for the null hypothesis of equal distribution, $P=Q$. `samesame` implements classifier two-sample tests (CTSTs) for this use case.

**Why CTSTs?** They are powerful, modular (flexible), and easy to explain—the elusive trifecta. For a deeper dive, see this [discussion](https://vathymut.org/posts/2022-01-22-in-gentle-praise-of-modern-tests/).

### How CTSTs Work

CTSTs leverage machine learning classifiers to test for distribution differences as follows:

1. **Label samples** - Assign class labels to indicate sample membership (`sample_P` = positive class, `sample_Q` = negative class)
2. **Train classifier** - Fit a classifier to predict sample labels based on features
3. **Evaluate performance** - If the classifier achieves high accuracy, the samples are likely different; if accuracy is near random chance, they are likely similar

This approach is versatile: you can use any classifier and any binary classification metric (AUC, balanced accuracy, etc.) without making strong distributional assumptions.

## Data

To demonstrate CTSTs in action, let's create a synthetic dataset with two distinguishable distributions:

```python
from sklearn.datasets import make_classification

# Create a small dataset with clear class separation
X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=123_456,
)
```

The two samples are concatenated into a single feature matrix `X` with shape `(100, 4)`. The binary labels `y` indicate sample membership—which observations belong to `sample_P` (positive class, `y=1`) and which belong to `sample_Q` (negative class, `y=0`).

## Cross-fitted Predictions

Cross-fitting estimates classifier performance on unseen data using out-of-sample predictions. This approach is more efficient than sample splitting, which typically uses 50% of data for training and 50% for testing—resulting in a loss of statistical power. Cross-fitting and out-of-bag methods use more data for inference while preserving statistical validity.

---
For more alternatives to sample splitting, see this [paper](https://www.cell.com/patterns/fulltext/S2666-3899(22)00237-9).

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import binarize

# Get cross-fitted probability predictions (out-of-sample)
y_gb = cross_val_predict(
    estimator=HistGradientBoostingClassifier(random_state=123_456),
    X=X,
    y=y,
    cv=10,
    method='predict_proba',
)[:, 1]  # Extract probabilities for the positive class

# Binarize predictions for metrics that require binary labels
y_gb_binary = binarize(y_gb.reshape(-1, 1), threshold=0.5).ravel()
```

## Classifier Two-Sample Tests (CTSTs)

Now we run CTSTs using three different classification metrics.

```python
from samesame.ctst import CTST
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

# Define metrics to evaluate
metrics = [balanced_accuracy_score, matthews_corrcoef, roc_auc_score]

# Run CTST for each metric
for metric in metrics:
    # Use binary predictions for metrics that require them; probabilities for AUC
    predicted = y_gb_binary if metric != roc_auc_score else y_gb
    ctst = CTST(actual=y, predicted=predicted, metric=metric)
    print(f"{metric.__name__}")
    print(f"\t statistic: {ctst.statistic:.2f}")
    print(f"\t p-value: {ctst.pvalue:.2f}")
```

**Output:**

```text
balanced_accuracy_score
     statistic: 0.86
     p-value: 0.00
matthews_corrcoef
     statistic: 0.72
     p-value: 0.00
roc_auc_score
     statistic: 0.93
     p-value: 0.00
```

In all three cases, we **reject the null hypothesis of equal distribution** ($P=Q$) at conventional significance levels. The `HistGradientBoostingClassifier` performed well across all metrics, providing strong evidence that `sample_P` and `sample_Q` come from different distributions.

## Out-of-Bag Predictions

An alternative to cross-fitting is using out-of-bag (OOB) predictions from ensemble methods like `RandomForestClassifier`. Both cross-fitted and OOB predictions
mitigate the downsides of sample splitting.

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest with out-of-bag score enabled
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=123_456,
    oob_score=True,
    min_samples_leaf=10,
)

# Get out-of-bag decision function scores (probabilities)
y_rf = rf.fit(X, y).oob_decision_function_[:, 1]

# Binarize for metrics requiring binary predictions
y_rf_binary = binarize(y_rf.reshape(-1, 1), threshold=0.5).ravel()

# Run CTST for each metric
for metric in metrics:
    predicted = y_rf_binary if metric != roc_auc_score else y_rf
    ctst = CTST(actual=y, predicted=predicted, metric=metric)
    print(f"{metric.__name__}")
    print(f"\t statistic: {ctst.statistic:.2f}")
    print(f"\t p-value: {ctst.pvalue:.2f}")
```

**Output:**

```text
balanced_accuracy_score
     statistic: 0.88
     p-value: 0.00
matthews_corrcoef
     statistic: 0.76
     p-value: 0.00
roc_auc_score
     statistic: 0.94
     p-value: 0.00
```

Again, we reject the null hypothesis of equal distribution. The `RandomForestClassifier` performed on par with the gradient boosting model, providing evidence of distribution shift. Both classifiers agree that the samples are distinguishable.

## Interpreting Results

**Important distinction:** A significant CTST indicates the distributions are *different*, but not necessarily that the difference is *problematic*. In production settings (e.g., model monitoring, data validation), you may want to use a [noninferiority test](../noninferiority.md) instead to determine if the shift is substantive enough to warrant action.

This example demonstrates the modularity of CTSTs:

- **Classifiers:** Any classifier works (here we used gradient boosting and random forests)
- **Metrics:** Any classification metric can be used (balanced accuracy, AUC, Matthews correlation coefficient, etc.)
- **Inference methods:** Both cross-fitting and out-of-bag predictions avoid the sample-splitting penalty

This flexibility makes CTSTs useful for diverse applications without compromising statistical power for valid inference.

## Explanations

To understand *why* the two samples are different, you can apply explainable methods to the fitted classifiers. For example, feature importance from `RandomForestClassifier` or model-agnostic techniques like SHAP can reveal which features contribute most to the distribution shift.

```python
# Example: Feature importance from random forest
import pandas as pd

importances = rf.feature_importances_
feature_names = [f"Feature {i}" for i in range(X.shape[1])]
importance_df = pd.DataFrame(
    {'Feature': feature_names, 'Importance': importances}
).sort_values('Importance', ascending=False)
print(importance_df)
```

This helps answer, *which features are driving the distribution shift?* This information is valuable for understanding root causes and deciding on remedial
actions.
