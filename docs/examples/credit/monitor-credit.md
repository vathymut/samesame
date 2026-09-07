# Monitor a credit model

An alarm should tell you whether a score moved toward worse outcomes, not just that it moved. Imagine a lender whose model was trained on a book of low-risk applicants. A new partner channel or a changing market brings in riskier applicants. The model is still scoring applications, and the shift could reflect model degradation, a change in the population, or both.

We start by describing the dataset and source-target comparison, then turn to the scores and the monitoring questions they can answer.

## The dataset

HELOC means **home equity line of credit**: a revolving credit line secured by the borrower's home. This dataset contains anonymized credit-bureau features from the FICO Explainable AI Challenge and a target indicating whether an account went 90 days past due or worse during its first 24 months. The modeling goal is to predict each applicant's probability of a bad outcome, often called the **default probability**; averaged across a group, those predictions give an expected **bad rate**. The example fetches 9,871 applications from [OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active) (`data_id=45554`) with `fetch_openml`.

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
    harm = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"Harm  p-value: {harm.pvalue:.4f}")   # → 0.0001
    ```

    The harmful-shift test points toward higher risk. Indeed, the mean predicted risk, or expected bad rate, rises from about 44% in source to 73% in target.

=== "Outlier score: confidence, no labels needed"

    [`LogitGap`](https://openreview.net/forum?id=FLdLPUqnsP) (Liang et al., 2025) measures how clearly the model favors one class over the other. A large gap means high confidence; a small gap means the model is undecided. Because lower confidence is harmful, we use `worse="lower"`.

    ```python
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
    --8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"
    train_conf = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
    deploy_conf = outlier_scores_from_probabilities(rf_bad.predict_proba(X_deployment))

    harm = ss.test_harmful_shift(
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

    harm = ss.test_harmful_shift(
        source=brier_source, target=brier_target,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"Brier p-value: {harm.pvalue:.4f}")  # -> 1.0000
    ```

    Default probability tells us what the model predicts; confidence tells us how sure it is. Brier score asks a different question: were those probabilities accurate once the outcomes arrived? With `p=1.0000`, we do not reject the null hypothesis of no harmful increase in prediction error. On this split, there is no evidence that the model's errors are larger in target. The higher risk appears to reflect a change in the population rather than material model deterioration.

## Wrap up

These three scores give three views of the same deployment change. Predicted risk tells us how harmful the target looks, confidence tells us how decisive the model is, and Brier score tells us whether its probabilities remain accurate once outcomes arrive. Here, the target looks riskier and the model is more confident, but prediction error does not increase. Together, the results suggest a riskier population rather than material model deterioration.

The next question is whether the comparison is being driven by regions where source and target have little overlap. Continue with [Weight for common support](weight-for-common-support.md) to make that question explicit.
