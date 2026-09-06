# Monitor a credit model

One HELOC split on ExternalRiskEstimate at 63 (Gardner et al., 2023) traces model degradation versus population change through three scores. You have one model and one split, with three ways to read harm.

| Signal | Needs labels? | Harmful direction | `worse` |
|--------|----------------|-------------------|---------|
| Predicted risk | No | Higher risk | `higher` |
| Outlier score — confidence (`LogitGap`) | No | Lower certainty | `lower` |
| Prediction error (Brier) | Yes | Larger error | `higher` |

--8<-- "snippets/source-target.txt"

Which signal to use depends on timing; [Which signal when?](#which-signal-when) compares them. New to `samesame`? Start with [Get started](../tutorials/get-started.md) or [Is the new drug good enough?](../trials/check-drug-efficacy.md).

## The dataset

HELOC (**home equity line of credit**): anonymized bureau features from the FICO Explainable AI Challenge (target: 90 days past due). Fetch 9,871 applications from [OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active) (`data_id=45554`) via `fetch_openml`.

## The split

`ExternalRiskEstimate` (higher is safer) at **63** is the FICO-winning and [TableShift](https://tableshift.org) split. Deployment story: 7,683 above 63 (**source**, calmer book, 43.5% bad) versus 2,188 at or below 63 (**target**, riskier deployment, 81.9% bad). Mean predicted risk about 44% versus 73%.

Same shift has two readings: model degraded or population changed. The signals below separate them; weighting ([Weight for common support](../../how-to/weight-for-common-support.md)) asks whether the alarm holds on common support. Setup:

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signals

=== "Risk — no labels needed"

    Predicted risk `P(default)`; larger means more harm, so `worse="higher"`. Risk and confidence share the 63-split (domain probabilities out-of-bag; see [Core concepts](../../explanation/core-concepts.md)).

    ```python
    import numpy as np
    import samesame as ss

    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"

    shift = ss.test_shift(
        source=domain_prob[split.values == 0],
        target=domain_prob[split.values == 1],
        rng=np.random.default_rng(12345),
    )
    harm = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")   # → 1.0000, 0.0002
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")    # → 0.2483, 0.0001
    ```

    AUC 1.00 is expected because the split variable is itself a feature. The harm test shows the shift points toward higher risk.

    | `test_shift` | `test_harmful_shift` | Interpretation |
    |--------------|----------------------|---------|
    | Significant | Significant | Changed and toward the harmful tail |
    | Significant | Not significant | Changed, not clearly harmful |
    | Not significant | Not significant | No clear shift |
    | Not significant | Significant | Tail signal missed by broad screen |

    Both tests point to higher risk. Investigate, don't auto-retrain. `0.5` is chance; read harm against its null and the 0–1 scale.

=== "Outlier score: confidence, no labels needed"

    `LogitGap` (gap between top logit and mean of the rest) is an **outlier score** for confidence. Larger means more certain, so a drop (`worse="lower"`) signals harm ([Core concepts](../../explanation/core-concepts.md)).

    ```python
    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"
    import samesame as ss

    train_conf = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
    deploy_conf = outlier_scores_from_probabilities(rf_bad.predict_proba(X_deployment))

    harm = ss.test_harmful_shift(
        source=train_conf, target=deploy_conf,
        worse="lower",  # lower confidence = harm
        rng=np.random.default_rng(12345),
    )
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")  # → 0.0409, 1.0000
    ```

    No harmful confidence drop. The statistic points the other way. At 82% bad rate predictions polarize and confidence rises. A model can grow more confident while predicting higher risk; confidence complements risk.

??? details "Errors — needs labels (under the null here)"

    Once labels arrive, test prediction error (Brier) with `worse="higher"`. In this guide the error section uses a separate **random** split under the null, so `p=0.2737` is expected. In deployment a small p-value would signal worse accuracy.

    ```python
    import samesame as ss

    harm = ss.test_harmful_shift(source=brier_train, target=brier_test, worse="higher", rng=np.random.default_rng(12345))
    print(f"Brier p={harm.pvalue:.4f}")  # → 0.2737 (no shift, as expected)
    ```

    Full script `examples/credit/_code/monitor_prediction_errors_example.py` loads HELOC, fits the RF out-of-sample, and computes Brier scores.

Full scripts: `examples/credit/_code/`.

## Which signal when?

| Signal | Needs labels? | When it helps |
|--------|----------------|----------------|
| Predicted risk | No | The output itself is harm |
| Outlier score — confidence | No | Early warning before labels arrive |
| Prediction error (Brier) | Yes | Clearest accuracy check once labels arrive |

Weighting answers whether the alarm reflects comparable applicants ([Weight for common support](../../how-to/weight-for-common-support.md); [Core concepts](../../explanation/core-concepts.md)). With one model and one split, choose the lens that matches whether labels have arrived.
