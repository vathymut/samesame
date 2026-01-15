# Test for Distribution Shifts

This is a concise walkthrough of classifier two-sample tests (CTST) to test whether two samples come from different distributions, using a small synthetic example. CTSTs are flexible (any classifier, any metric) and require few assumptions.

## Data

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=123_456,
)
# y = 1 denotes sample_P, y = 0 denotes sample_Q
```

## Cross-fitted CTST (recommended)

Use cross-fitted predictions to avoid sample-splitting and get valid p-values.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from samesame.ctst import CTST
from sklearn.metrics import roc_auc_score

# Out-of-sample predicted probabilities
y_hat = cross_val_predict(
    HistGradientBoostingClassifier(random_state=123_456),
    X,
    y,
    cv=10,
    method="predict_proba",
)[:, 1]

# Run CTST with AUC
ctst = CTST(actual=y, predicted=y_hat, metric=roc_auc_score)
print("CTST (AUC)")
print(f"  statistic: {ctst.statistic:.2f}")
print(f"  p-value:   {ctst.pvalue:.4f}")
```

**Output:**

```text
CTST (AUC)
    statistic: 0.93
    p-value:   0.0002
```

**Interpretation**: A small p-value rejects $P=Q$, indicating the samples differ. If p-value is large, evidence is insufficient to claim a shift.

## OOB Alternative

Out-of-bag (OOB) predictions from ensembles can replace cross-fitting when convenient.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    min_samples_leaf=10,
    random_state=123_456,
)
rf.fit(X, y)
y_oob = rf.oob_decision_function_[:, 1]

ctst_oob = CTST(actual=y, predicted=y_oob, metric=roc_auc_score)
print("CTST (OOB, AUC)")
print(f"  statistic: {ctst_oob.statistic:.2f}")
print(f"  p-value:   {ctst_oob.pvalue:.4f}")
```

**Output:**

```text
CTST (OOB, AUC)
    statistic: 0.94
    p-value:   0.0002
```

## Interpreting Results

- Small p-value → distributions differ (shift detected).
- Large p-value → insufficient evidence of shift.
- CTST says *whether* distributions differ, not *why* or *how bad*; pair with DSOS for adverse shift checks.

## Tips

- Use cross-fitting or OOB predictions to boost statistical power and avoid training bias.
- Pick a metric aligned with your classifier output (AUC for probabilities; balanced accuracy for binary predictions).
- For explanations, inspect feature importance or SHAP on the CTST classifier to see what drives the shift.
- For adverse-shift questions, continue to [Noninferiority](noninferiority.md) or apply DSOS on relevant scores.
