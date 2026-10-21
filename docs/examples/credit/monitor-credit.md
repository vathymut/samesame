# Monitor a credit model

An alarm should tell you whether a score moved toward worse outcomes, not just that it moved. Imagine a lender whose model was trained on a book of low-risk applicants. A new partner channel, or a changing market, brings in riskier applicants. The model is still scoring applications. The shift could be model degradation, a change in the population, or both.

## The dataset

HELOC means **home equity line of credit**: a revolving credit line secured by the borrower's home. This dataset, from the FICO Explainable AI Challenge, has anonymized credit-bureau features and a label for whether the account went 90 days past due, or worse, in its first 24 months. The model predicts each applicant's probability of that bad outcome, the **default probability**. Averaged over a group, those predictions are an expected **bad rate**. The example loads 9,871 applications from [OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active) (`data_id=45554`) with `fetch_openml`.

## The split

To make the comparison concrete, we split applications at **63** on `ExternalRiskEstimate`, following Gardner et al. (2023). Higher values indicate safer applicants, so applications above 63 form the lower-risk source group, while applications at or below 63 form the higher-risk target group:

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
| Outlier score: confidence (`LogitGap`) | No | Lower certainty | `lower` | Early warning before labels arrive |
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

    The harmful-shift test points toward higher risk. The mean predicted risk, the expected bad rate, rises from about 44% in source to 73% in target.

=== "Outlier score: confidence, no labels needed"

    [`LogitGap`](https://openreview.net/forum?id=FLdLPUqnsP) (Liang et al., 2025) measures how clearly the model favors one class over the other. A large gap means high confidence; a small gap means the model is undecided. Because lower confidence is harmful, we use `worse="lower"`.

    ```python
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"
    train_conf = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
    deploy_conf = outlier_scores_from_probabilities(rf_bad.predict_proba(X_deployment))

    harm = ss.test_harm(
        source=train_conf, target=deploy_conf,
        worse="lower",  # lower confidence = harm
        rng=np.random.default_rng(12345),
    )
    print(f"Harm  p-value: {harm.pvalue:.4f}")  # → 1.0000
    ```

    Default probability and confidence answer different questions. Default probability asks, "How risky does the model think this applicant is?" `LogitGap` asks, "How sure is it?" Here, target predictions are both riskier and more confident, so there is no harmful confidence drop. The model is more decisive, not safer: confidence adds a second dimension to the risk score.

=== "Prediction error, labels needed"

    Once labels arrive, prediction error (Brier) gives the clearest post-outcome check. It uses the same source and target split and the same source-trained model as the other scores; the difference is that labels are now available to measure each prediction's error.

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

    Brier score asks whether those probabilities were accurate once the outcomes arrived. With `p=1.0000`, we do not reject the null of no harmful increase in prediction error. On this split, the higher risk looks like a change in the population, not larger model errors.

## Wrap up

Predicted risk rose, confidence did not fall, and prediction error did not rise. On this split the target looks like a riskier book, not a model that got worse at its probabilities.

If a few low-overlap applications are driving that comparison, [Weight for common support](weight-for-common-support.md) is the next check.
