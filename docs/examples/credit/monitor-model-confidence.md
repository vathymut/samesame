# How to: Monitor model confidence

Use this when you want a certainty signal alongside business risk — or when the model output isn't what you want to monitor.

This guide uses the same HELOC split as [Monitor predicted credit risk](monitor-credit-risk.md) — **source** = training (lower-risk), **target** = deployment (higher-risk) — but asks a different question.

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

print(f"Source (training) mean confidence:   {train_confidence.mean():.3f}")
print(f"Target (deployment) mean confidence: {deployment_confidence.mean():.3f}")
```

On this split, target confidence is higher than source confidence across the distribution.

## Step 3 — Did confidence drop?

Higher confidence is better, so a harmful shift is toward *lower* scores.

```python
harm = ss.test_harmful_shift(
    source=train_confidence,
    target=deployment_confidence,
    worse="lower",  # lower confidence = harm (or ss.Worse.LOWER)
    rng=np.random.default_rng(12345),
)

print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
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
    import re

    import numpy as np
    import pandas as pd
    from scipy.special import logit
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier

    import samesame as ss

    def logit_gap(logits: np.ndarray) -> np.ndarray:
        max_logits = np.max(logits, axis=1)
        mean_rest = (np.sum(logits, axis=1) - max_logits) / (logits.shape[1] - 1)
        return max_logits - mean_rest

    def outlier_scores_from_probabilities(probabilities, *, clip=1e-6):
        clipped = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
        return logit_gap(logit(clipped))

    # --- HELOC split (same as credit-risk guide)
    fico = fetch_openml(data_id=45554, as_frame=True)
    X, y = fico.data, fico.target
    re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
    col_split = next((c for c in X.columns if re_obj.search(c)), None)
    mask_high = X[col_split].astype(float) > 63
    X_train = X[mask_high].reset_index(drop=True)
    y_train = y[mask_high].reset_index(drop=True)
    X_deployment = X[~mask_high].reset_index(drop=True)

    # --- Train credit model (for confidence, not risk)
    y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values
    rf_bad = RandomForestClassifier(
        n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
    )
    rf_bad.fit(X_train, y_train_binary)

    # --- Build confidence (outlier) scores — larger = more certain
    train_confidence = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
    deployment_confidence = outlier_scores_from_probabilities(
        rf_bad.predict_proba(X_deployment)
    )
    print(f"Training mean confidence:   {train_confidence.mean():.3f}")
    print(f"Deployment mean confidence: {deployment_confidence.mean():.3f}")

    harm = ss.test_harmful_shift(
        source=train_confidence,
        target=deployment_confidence,
        worse="lower",
        rng=np.random.default_rng(12345),
    )
    print(f"Statistic: {harm.statistic:.4f}")
    print(f"p-value:   {harm.pvalue:.4f}")
    ```

    Training scores use `oob_decision_function_` (honest, out of sample). See
    [Glossary](../../explanation/glossary.md#honest-out-of-sample-scores).

## Next steps

- If labels are available, see [Monitor prediction errors once labels arrive](monitor-prediction-errors.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? note "Why `outlier_scores_from_probabilities`?"
    The package calls anomaly-like scalars **outlier scores**. Confidence fits that recipe: an outlier score where higher = more typical. The function name reflects the general recipe, not just confidence.
