# How to: Monitor model confidence

Use this when you want a certainty signal alongside business risk — or when the model output isn't what you want to monitor.

This guide uses the same HELOC split as [Monitor predicted credit risk](monitor-credit-risk.md) — **source** = training (lower-risk), **target** = deployment (higher-risk) — but asks a different question.

!!! info "Prerequisites"
    - [Monitor predicted credit risk](monitor-credit-risk.md) — same HELOC split and OOB workflow, but for risk.
    - [Test whether the shift is harmful](../tutorials/check-shift-harm.md) — `worse` and direction.

## Why confidence is separate

Predicted risk and model confidence are not the same.

- **Predicted risk** — does the model think outcomes are worse?
- **Confidence** — is the model more or less certain?

They can move independently. A model can become more confident while predicting riskier outcomes.

In `samesame` terms, confidence is an **outlier score** where higher = more in-distribution / more certain. See [Glossary: Score](../../explanation/glossary.md#score). The helper `outlier_scores_from_probabilities` follows that naming — larger means higher confidence.

## Setup

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

--8<-- "snippets/honest-scores-ref.txt"

## Step 1 — Train the model

```python
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

## Step 2 — Build a confidence (outlier) score

We use `LogitGap` — the gap between the top logit and the mean of the rest. Larger gaps mean stronger class separation, interpreted as higher confidence.

```python
import numpy as np
import samesame as ss

--8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
--8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
--8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"

train_probabilities = rf_bad.oob_decision_function_  # source, out-of-sample
deployment_probabilities = rf_bad.predict_proba(X_deployment)  # target

train_confidence = outlier_scores_from_probabilities(train_probabilities)
deployment_confidence = outlier_scores_from_probabilities(deployment_probabilities)

print(f"Source (training) mean confidence:   {train_confidence.mean():.3f}")  # → ~1.2
print(f"Target (deployment) mean confidence: {deployment_confidence.mean():.3f}")  # → ~1.5 (higher)
```

On this split, target confidence is higher than source confidence across the distribution.

## Step 3 — Did confidence drop?

Higher confidence is better, so a harmful shift is toward *lower* scores.

```python
harm = ss.test_harmful_shift(
    source=train_confidence,
    target=deployment_confidence,
    worse="lower",  # lower confidence = harm
    rng=np.random.default_rng(12345),
)

print(f"Statistic: {harm.statistic:.4f}")  # → ~0.04
print(f"p-value:   {harm.pvalue:.4f}")     # → ~0.90 (large)
```

Expect a large p-value — confidence shifts toward *higher* scores here, so no harmful drop.

## Read the result

--8<-- "snippets/pvalue-guidance.txt"

This doesn't contradict the credit-risk guide — it answers a different question:

- In [Monitor predicted credit risk](monitor-credit-risk.md), predicted default risk rises.
- Here, confidence also rises — the model is not less certain on target.

A model can be confidently risky, confidently wrong, or confidently stable. Confidence is context, not a substitute for a business signal.

??? example "Full script — copy and run"
    ```python
    --8<-- "examples/credit/_code/monitor_model_confidence_full.py:full"
    ```

    Training scores use `oob_decision_function_` (honest, out of sample). See
    [Glossary](../../explanation/glossary.md#honest-out-of-sample-scores).

## Next steps

- If labels are available, see [Monitor prediction errors once labels arrive](monitor-prediction-errors.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? note "Why `outlier_scores_from_probabilities`?"
    The package calls anomaly-like scalars **outlier scores**. Confidence fits that recipe: an outlier score where higher = more typical. The function name reflects the general recipe, not just confidence.
