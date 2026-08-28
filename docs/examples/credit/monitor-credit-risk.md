# How to: Monitor predicted credit risk

Use this when your model output already has business meaning and you need two answers: did the target shift, and did predicted risk get worse?

!!! info "Prerequisites"
    - [Detect any distributional shift](../tutorials/detect-distribution-shift.md) — your first shift test.
    - Familiarity with `predict_proba` / `RandomForestClassifier`. New to `samesame`? Start with the tutorials first.

## Why this signal

Predicted default probability is directly interpretable — larger is worse — so it fits `ss.test_harmful_shift(..., worse="higher")`. See [Glossary: `worse`](../../explanation/glossary.md#worse). Strings and `ss.Worse` enum are interchangeable.

This guide uses the HELOC dataset. **Source** = lower-risk customers (training); **Target** = higher-risk customers (simulated deployment). From here we use **source/target** consistently — see [Glossary](../../explanation/glossary.md#source-and-target).

## Setup

The HELOC split is shared across the credit guides. We include it as a snippet so all guides stay in sync.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

`fetch_openml` needs internet. Cache the dataset locally if you run offline.

--8<-- "snippets/honest-scores-ref.txt"

## Step 1 — Is the target different?

Train a domain classifier to distinguish source from target. Source observations are scored out-of-sample via `oob_decision_function_` (bagged forests). For other estimators, use `cross_val_predict` as in the [first tutorial](../tutorials/detect-distribution-shift.md).

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-domain"

shift = ss.test_shift(
    source=domain_prob[split.values == 0],
    target=domain_prob[split.values == 1],
    rng=np.random.default_rng(12345),
)

print(f"AUC statistic: {shift.statistic:.4f}")  # → 0.9999
print(f"p-value:       {shift.pvalue:.4f}")     # → 0.0001
```

On this split expect AUC ≈ `1.0` and a very small p-value — source and target separate clearly.

Quick diagnostic — what changed:

```python
feature_importance = (
    pd.Series(rf_domain.feature_importances_, index=X_concat.columns)
    .sort_values(ascending=False)
)

print(feature_importance.head(5))
```

## Step 2 — Did predicted risk rise?

Train the credit model. Use out-of-bag predictions for source and standard predictions for target.

```python
--8<-- "snippets/heloc-split.py:heloc-risk-model"

harm = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",  # larger = worse
    rng=np.random.default_rng(12345),
)

print(f"Statistic: {harm.statistic:.4f}")  # → ~0.35
print(f"p-value:   {harm.pvalue:.4f}")     # → 0.0001
```

Expect `p < 0.001` — target not only differs but carries higher predicted risk.

## Step 3 — Read the result

--8<-- "snippets/pvalue-guidance.txt"

| `test_shift` | `test_harmful_shift` | What it usually means |
|--------------|----------------------|-----------------------|
| significant (p ≤ 0.05) | significant | population changed **and** predicted risk worsened |
| significant | not significant | population changed, but not in a clearly harmful way |
| not significant | significant | rare — directional signal is strong where two-sided is not; investigate |
| not significant | not significant | no clear evidence of harmful shift |

On this HELOC split both tests are significant — consider retraining, recalibration, or a deployment policy change.

??? example "Full script — copy and run"
    ```python
    --8<-- "examples/credit/_code/monitor_credit_risk_example.py:full"
    ```

    Scores from a fitted model must be out of sample — here `oob_decision_function_` for
    training rows and `predict_proba` for deployment rows. See [Glossary](../../explanation/glossary.md#honest-out-of-sample-scores).

## Next steps

- For a certainty view, see [Monitor model confidence](monitor-model-confidence.md).
- If labels are available, see [Monitor prediction errors once labels arrive](monitor-prediction-errors.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? tip "Why OOB here?"
    OOB predictions are free for `RandomForestClassifier(oob_score=True)` and out-of-sample by construction. For `HistGradientBoostingClassifier` or other estimators, use `cross_val_predict(cv=10, method="predict_proba")` — same interpretation, different estimator.
