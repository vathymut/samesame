# How to: Monitor model confidence

Use this guide when you want a certainty signal alongside business risk, or when the model output itself is not the thing you want to monitor.

This guide uses the same HELOC setup as [Monitor predicted credit risk](monitor-credit-risk.md), but asks a different question.

## Why confidence is a separate signal

Predicted risk and model confidence are not the same thing.

- **Predicted risk** asks whether the model thinks outcomes are worse.
- **Confidence** asks whether the model is more or less certain.

Those signals can move together or independently. A model can become more confident while still predicting riskier outcomes.

In `samesame` terminology, confidence is treated as an **outlier score** where higher means "more in-distribution / more certain." The helper below is called `outlier_scores_from_probabilities` for that reason — here larger scores mean higher confidence.

## Setup

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

--8<-- "snippets/honest-scores.txt"

## Step 1 — Train the model

```python
import samesame as ss
from sklearn.ensemble import RandomForestClassifier

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

## Step 2 — Build a confidence (outlier) score from class probabilities

We use `LogitGap`, the gap between the top logit and the mean of the rest. Larger gaps mean stronger class separation — interpreted as higher confidence.

```python
import numpy as np

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

On this HELOC split, deployment confidence is higher than training confidence across the distribution.

## Step 3 — Test whether confidence dropped

Higher confidence is better, so a harmful shift is toward *lower* scores.

```python
source_scores = train_confidence
target_scores = deployment_confidence

harm = ss.test_harmful_shift(
    source=source_scores,
    target=target_scores,
    worse="lower",  # lower confidence = harm
    rng=np.random.default_rng(12345),
)

print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
```

Expect a large p-value here — deployment confidence shifts toward *higher* scores, so we do not flag a harmful confidence drop.

## What this tells you

This does not contradict the credit-risk guide. It answers a different question.

- In [Monitor predicted credit risk](monitor-credit-risk.md), predicted default risk rises.
- Here, confidence also rises — the model does not look less certain on deployment.

That combination is entirely possible: a model can be confidently risky, confidently wrong, or confidently stable. Confidence is useful context, but not a substitute for a business signal when one exists. If labels are available, see [Monitor prediction errors once labels arrive](monitor-prediction-errors.md).

??? note "Why `outlier_scores_from_probabilities`?"
    The package calls anomaly-like scalars **outlier scores** (`CONTEXT.md` terminology). Confidence fits that recipe: it is an outlier score where higher = more typical. The function name reflects the general recipe, not just this confidence use-case.
