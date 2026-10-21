# Monitor a credit model

An alarm should tell you whether a score moved toward worse outcomes, not just that it changed. Imagine a lender whose model was trained on low-risk applicants. A new partner channel or a changing market brings in riskier applicants. The model keeps scoring applications, but the shift could reflect a worse model, a different population, or both.

## The dataset

HELOC means **home equity line of credit**: a revolving credit line secured by
the borrower's home. This FICO Explainable AI Challenge dataset contains
anonymized credit-bureau features. It also records whether an account went
90 days past due, or worse, in its first 24 months.

The model estimates each applicant's probability of that outcome (**default
probability**). A group's average prediction is its expected **bad rate**.
The example loads 9,871 applications from [OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active)
(`data_id=45554`) with `fetch_openml`.

## The split

We split applications at **63** on `ExternalRiskEstimate`, following Gardner et al. (2023). Higher values indicate safer applicants: those above 63 form the lower-risk source group, while those at or below 63 form the higher-risk target group:

- **Source:** 7,683 applications above 63, the low-risk book, with a 43.5% observed bad rate.
- **Target:** 2,188 applications at or below 63, the riskier book, with an 81.9% observed bad rate.

The code below creates these source and target samples:

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Scores

With source and target defined, we can compare three scores. They are not interchangeable: each answers a different question and becomes available at a different time.

| Score | Requires labels? | Harmful direction | `worse` | When it helps |
|--------|-------------------|-------------------|---------|---------------|
| Predicted risk | No | Higher risk | `higher` | The output itself represents harm |
| Confidence (`LogitGap`) | No | Lower certainty | `lower` | Early warning before labels arrive |
| Prediction error (Brier) | Yes | Larger error | `higher` | Post-outcome accuracy check |

Use predicted risk when the model output already represents harm, confidence for an early warning while labels are delayed, and prediction error for the clearest post-outcome check.

The score examples build on the same model and out-of-sample predictions:

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-risk-model"
```

=== "Risk, no labels needed"

    Start with predicted risk when the model output already represents harm. `P(default)` is directly tied to the outcome: larger means more harm, so use `worse="higher"`.

    ```python
    harm = ss.test_harm(
        source=train_risk, target=deployment_risk,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"Harm  p-value: {harm.pvalue:.4f}")   # → 0.0001
    ```

    The test finds evidence of movement toward higher risk. Mean predicted risk—the expected bad rate—rises from about 44% in source to 73% in target.

=== "Confidence score, no labels needed"

    [`LogitGap`](https://openreview.net/forum?id=FLdLPUqnsP) (Liang et al., 2025)
    measures how clearly the model favors one class over the other. A large gap
    means high confidence; a small gap means the model is undecided. Lower
    confidence is harmful here, so we use `worse="lower"`.

    ```python
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:confidence-scores"
    train_conf = confidence_scores_from_probabilities(
        rf_bad.oob_decision_function_
    )
    deploy_conf = confidence_scores_from_probabilities(
        rf_bad.predict_proba(X_deployment)
    )

    harm = ss.test_harm(
        source=train_conf, target=deploy_conf,
        worse="lower",  # lower confidence = harm
        rng=np.random.default_rng(12345),
    )
    print(f"Harm  p-value: {harm.pvalue:.4f}")  # → 1.0000
    ```

    Default probability and confidence answer different questions. The first asks,
    "How risky does the model think this applicant is?" `LogitGap` asks, "How
    sure is the model?" Target predictions are both riskier and more confident.
    The test finds no evidence of a harmful confidence drop. Higher confidence
    alone does not mean the model is safer; confidence adds a separate dimension
    to risk.

=== "Prediction error, labels needed"

    Once labels arrive, prediction error (Brier) gives a post-outcome check. It
    uses the same split and source-trained model as the other scores. Now that
    labels are available, we can measure each prediction's error.

    ```python
    y_deployment_binary = y_deployment.map({"Good": 0, "Bad": 1}).astype(int).values
    brier_source = (y_train_binary - train_risk) ** 2
    brier_target = (y_deployment_binary - deployment_risk) ** 2

    harm = ss.test_harm(
        source=brier_source, target=brier_target,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"Brier p-value: {harm.pvalue:.4f}")  # → 1.0000
    ```

    Brier score asks whether those probabilities were accurate once outcomes
    arrived. With `p=1.0000`, we do not reject the null of no harmful increase
    in prediction error. Risk rises on this split, but the test finds no
    evidence of a harmful error increase.

## Wrap up

Predicted risk is higher in target, while these tests find no evidence of
lower confidence or higher prediction error. This pattern is consistent with
a riskier population. This split cannot establish why risk increased or prove
that model quality is unchanged.

If a few low-overlap applications are driving that comparison, [Weight for common support](weight-for-common-support.md) is the next check.
