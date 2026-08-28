# Honest (out-of-sample) scores

Scores from a fitted model must be **out of sample** — `cross_val_predict`, `oob_decision_function_`, or held-out set. In-sample predictions create false separation and invalidate the test.

`samesame` only sees scores, not how they were made, so it cannot check this for you.

**Recipe — pick one:**

```python
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier

# CV (any estimator with predict_proba)
scores = cross_val_predict(model, X, y, cv=10, method="predict_proba")[:, 1]

# OOB (RandomForest with oob_score=True) — free
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345)
rf.fit(X_train, y_train)
train_scores = rf.oob_decision_function_[:, 1]   # honest, out-of-sample
deployment_scores = rf.predict_proba(X_deployment)[:, 1]  # held-out is honest
```

Failure mode: if you fit on `X_train` and score the same `X_train` with `predict_proba`, source and target separate even under the null — p-values are meaningless.

See [Detect any distributional shift](../../examples/tutorials/detect-distribution-shift.md) (CV) and [Monitor predicted credit risk](../../examples/credit/monitor-credit-risk.md) (OOB).
