# How to: Monitor model confidence

Use this guide when you want a confidence signal alongside business risk, or when the model output
itself is not the main thing you want to monitor.

This guide uses the same HELOC setup as [Monitor predicted credit risk](monitor-credit-risk.md),
but asks a different question.

## Why confidence is a separate signal

Predicted risk and model confidence are not the same thing.

- **Predicted risk** asks whether the model thinks outcomes are worse.
- **Confidence** asks whether the model looks more or less certain about its predictions.

Those signals can move together, but they do not have to. A model can become more confident while
still producing riskier outcomes.

## Setup

```python
import re

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
```

## Step 1 - Train the model

```python
bad_mapping = {"Good": 0, "Bad": 1}
y_train_binary = y_train.map(bad_mapping).values

rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)
```

## Step 2 - Build a confidence score from class probabilities

This example uses `LogitGap`, a small helper built from logit-transformed class probabilities.
Higher values mean the model separates the classes more strongly, which we treat as higher
confidence.

```python
--8<-- "_code/monitor_model_confidence_example.py:imports"
--8<-- "_code/monitor_model_confidence_example.py:logit-gap"
--8<-- "_code/monitor_model_confidence_example.py:outlier-scores"

train_probabilities = rf_bad.oob_decision_function_
deployment_probabilities = rf_bad.predict_proba(X_deployment)

train_confidence = outlier_scores_from_probabilities(train_probabilities)
deployment_confidence = outlier_scores_from_probabilities(deployment_probabilities)

print(f"Training mean confidence:   {train_confidence.mean():.3f}")
print(f"Deployment mean confidence: {deployment_confidence.mean():.3f}")
```

On this HELOC split, deployment confidence is higher on average than training confidence.

## Step 3 - Test whether confidence dropped

Higher confidence is better, so use `direction="higher-is-better"`.

```python
source_scores = train_confidence
target_scores = deployment_confidence

harm = ss.shift.detect_harm(
    source=source_scores,
    target=target_scores,
    direction="higher-is-better",
    random_state=12345,
)

print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
```

This workflow should not flag a harmful confidence shift on the HELOC split, because deployment
confidence moves up rather than down.

## What this tells you

This does not contradict the credit-risk guide. It answers a different question.

- In [Monitor predicted credit risk](monitor-credit-risk.md), predicted default risk rises.
- Here, confidence also rises, so the model does not look less certain on deployment.

That combination is entirely possible. A model can look confidently wrong, confidently risky, or
confidently stable. Confidence is useful context, but it is not a substitute for a business signal
when a business signal already exists.

If labels are available, the next step is often
[Monitor prediction errors once labels arrive](monitor-prediction-errors.md).