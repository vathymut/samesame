# How to: Monitor predicted credit risk

Use this guide when the model output already has business meaning and you want two answers: does deployment look different from training, and is predicted default risk higher?

If you are new to `samesame`, start with the tutorials first. This guide assumes familiarity with `predict_proba`.

## Why this signal works well

Predicted default probability is directly interpretable — larger values are worse — so it is a natural signal for `ss.test_harmful_shift(..., worse="higher")` (or `ss.Worse.HIGHER`).

This guide uses the HELOC dataset and simulates deployment by training on lower-risk customers and testing on higher-risk ones.

## Setup

The HELOC split is shared across the credit guides. We include it as a snippet so all guides stay in sync.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

`fetch_openml` needs internet access. Cache the dataset locally if you run offline.

--8<-- "snippets/honest-scores.txt"

## Step 1 — Check whether deployment looks different

Train a domain classifier to distinguish training from deployment. Each training observation is scored out of sample via `oob_decision_function_` (bagged forests). For other classifiers, use `cross_val_predict` as in the [first tutorial](../tutorials/detect-distribution-shift.md).

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-domain"

shift = ss.test_shift(
    source=domain_prob[split.values == 0],
    target=domain_prob[split.values == 1],
    rng=np.random.default_rng(12345),
)

print(f"AUC statistic: {shift.statistic:.4f}")
print(f"p-value:       {shift.pvalue:.4f}")
```

On this split you should see AUC close to `1.0` and a very small p-value — deployment looks clearly different from training.

Quick diagnostic — what changed:

```python
feature_importance = (
    pd.Series(rf_domain.feature_importances_, index=X_concat.columns)
    .sort_values(ascending=False)
)

print(feature_importance.head(5))
```

## Step 2 — Check whether predicted risk increased

Now train the actual credit model. Use out-of-bag predictions for training and standard predictions for deployment.

```python
--8<-- "snippets/heloc-split.py:heloc-risk-model"

harm = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",  # larger = worse (or ss.Worse.HIGHER)
    rng=np.random.default_rng(12345),
)

print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
```

Expect a very small p-value — deployment not only differs but carries higher predicted risk.

## Step 3 — Interpret the result

A small p-value (typically ≤ 0.05) is evidence against the null. Start with `.pvalue` for evidence; use `.statistic` for magnitude.

| `test_shift` | `test_harmful_shift` | What it usually means |
|--------------|----------------------|-----------------------|
| significant (p ≤ 0.05) | significant | population changed **and** predicted risk worsened |
| significant | not significant | population changed, but not in a clearly harmful way |
| not significant | significant | rare — directional signal is strong where two-sided is not, investigate directly |
| not significant | not significant | no clear evidence of harmful shift |

On this HELOC split both tests are significant — investigate retraining, recalibration, or a deployment policy change.

## Next steps

- For a separate certainty view, see [Monitor model confidence](monitor-model-confidence.md).
- If labels are available, see [Monitor prediction errors once labels arrive](monitor-prediction-errors.md).
- If overlap is poor, see [Focus harmful-shift testing on common support](../weighting/source-reweighting.md).

??? tip "Why OOB here?"
    OOB predictions are free for `RandomForestClassifier(oob_score=True)` and are out-of-sample by construction. For `HistGradientBoostingClassifier` or other estimators, use `cross_val_predict(cv=10, method="predict_proba")` — same interpretation, different estimator.
