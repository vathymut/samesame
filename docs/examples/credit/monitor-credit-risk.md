# How to: Monitor predicted credit risk

Use this guide when the model output already has clear business meaning and you want to know two
things: whether deployment looks different from training, and whether predicted default risk is
higher in deployment.

If you are new to `samesame`, start with the two tutorials first. This guide assumes basic
familiarity with fitting a scikit-learn classifier and calling `predict_proba(...)`.

## Why this signal works well

Predicted default probability is already meaningful. Larger values are directly worse, so it is a
natural signal for `ss.test_harmful_shift(...)`.

This guide uses the HELOC dataset and simulates deployment by training on lower-risk customers and
testing on higher-risk customers.

## Setup

```python
import re

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

import samesame as ss

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)

print(f"Training set:   {len(X_train)} samples")
print(f"Deployment set: {len(X_deployment)} samples")
```

## Step 1 - Check whether deployment looks different

Train a domain classifier to distinguish training from deployment. Use out-of-bag predictions so
each training observation is scored by trees that did not train on it.

```python
split = pd.Series([0] * len(X_train) + [1] * len(X_deployment))
X_concat = pd.concat([X_train, X_deployment], ignore_index=True)

rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
domain_scores = rf_domain.oob_decision_function_[:, 1]

shift = ss.test_shift(
    source=domain_scores[split.values == 0],
    target=domain_scores[split.values == 1],
)

print(f"AUC statistic: {shift.statistic:.4f}")
print(f"p-value:       {shift.pvalue:.4f}")
```

On this split, you should see an AUC close to `1.0` and a very small p-value, which means the
deployment population looks clearly different from training.

If you want a quick diagnostic on what changed, inspect the same classifier's feature importances:

```python
feature_importance = (
    pd.Series(rf_domain.feature_importances_, index=X_concat.columns)
    .sort_values(ascending=False)
)

print(feature_importance.head(5))
```

## Step 2 - Check whether predicted risk increased

Now train the actual credit model on the training set. Use out-of-bag predictions for training and
standard predictions for deployment.

```python
loan_status = y_train.map({"Good": 0, "Bad": 1}).values

rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, loan_status)

train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()

harm = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
)

print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
```

On this split, you should again see a very small p-value. That means deployment is not only
different, but also shows higher predicted risk according to the model.

## Step 3 - Decide what to do

Using both tests together gives a clearer picture than either test alone. Call a result
**significant** when its p-value is small (typically `<= 0.05`); otherwise call it **not significant**.

| `test_shift` | `test_harmful_shift` | What it usually means |
|----------------|----------------|------------------------|
| significant | significant | the population changed **and** predicted risk worsened |
| significant | not significant | the population changed, but not in a clearly harmful way |
| not significant | significant | rare; the directional signal is strong where the two-sided one is not — investigate directly |
| not significant | not significant | no clear evidence of harmful shift |

In this HELOC example, both tests are significant. These results justify investigating retraining,
recalibration, or changes to the deployment policy for the new population.

If you want a separate confidence view, see [Monitor model confidence](monitor-model-confidence.md).
If labels are available, see
[Monitor prediction errors once labels arrive](monitor-prediction-errors.md).
