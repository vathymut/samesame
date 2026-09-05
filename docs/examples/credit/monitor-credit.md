# Monitor a credit model

A useful alarm distinguishes whether a score moved at all from whether it moved toward worse outcomes. A lender that trained on its safest applicants shows why that distinction matters.

A lender trains a default-risk model on its most favorable book — the applicants the bureau already views as safe. Then the book changes: a new partner channel, a broader marketing push, or a market downturn brings riskier applicants. The model keeps scoring, and two explanations for the same shift look identical from the outside: the model may have degraded, or the population may have changed. The first suggests retraining; the second suggests a business decision. Using one HELOC split — ExternalRiskEstimate above 63 versus at or below 63 (Gardner et al., 2023) — this example follows that story through three scores from the same model. They answer different questions and become available at different times.

| Signal | Needs labels? | Harmful direction | `worse` |
|--------|----------------|-------------------|---------|
| Predicted risk | No | Higher risk | `higher` |
| Outlier score — confidence (`LogitGap`) | No | Lower certainty | `lower` |
| Prediction error (Brier) | Yes | Larger error | `higher` |

--8<-- "snippets/source-target.txt"

Which signal to lead with depends on your setting; [Which signal when?](#which-signal-when) compares them. If you are new to `samesame`, start with [Get started](../tutorials/get-started.md) or with [Is the new drug good enough?](../trials/check-drug-efficacy.md), where the same harmful-shift test is shown as a clinical trial with no model.

## The dataset

HELOC stands for **home equity line of credit**: a revolving credit line secured by the borrower's home. The data come from the FICO Community Explainable AI Challenge — anonymized credit-bureau features for each applicant, plus a target that records whether the borrower was 90 days past due or worse at least once in the first 24 months. We fetch the 9,871 applications from the [OpenML](https://openml.org/search?type=data&sort=runs&id=45554&status=active) mirror (`data_id=45554`) with `fetch_openml`, so you can run the example without a manual download.

## The split

Each HELOC applicant arrives with a third-party bureau risk estimate, the `ExternalRiskEstimate` feature, where higher values mean safer applicants. The winning model from the original FICO competition focused on a cutoff of **63** on this estimate, and the [TableShift](https://tableshift.org) benchmark (Gardner et al., 2023) adopted the same threshold to define its two populations. We follow that split and read it as a deployment story:

- **Source** — 7,683 applicants with an estimate above 63. This is the calmer book the lender trained on. The observed bad rate is 43.5%.
- **Target** — 2,188 applicants with an estimate at or below 63. This is the riskier book that arrives after deployment. The observed bad rate is 81.9%.

A model trained on the first group is then evaluated on the second: trained in calm conditions and tested in a more adverse mix. Mean predicted default risk rises from about 44% to 73%, and the two explanations return: has the model worsened, or has the context changed? The three signals below provide the evidence; the [weighting guide](../weighting/weight-for-common-support.md) then asks whether the alarm holds on common support.

## Setup

The same HELOC split (source: lower-risk training data; target: higher-risk deployment) is reused for risk and confidence. The error tab uses a separate random train-test split where the null of no error shift holds, so a large p-value is the expected result.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signals

=== "Risk — no labels needed"

    Predicted risk is the model's estimate of `P(default)`; larger values mean more harm, so `worse="higher"`. Declare the direction before testing and do not choose it by p-value.

    ```python
    import numpy as np
    import samesame as ss

    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"

    # any shift? domain_prob is generic; harm is tested on interpretable risk
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

    The domain classifier separates the two books almost perfectly (AUC 1.00), which is expected because the split variable itself is a feature. The harm test confirms that the shift points toward higher risk rather than just any change.

    | `test_shift` | `test_harmful_shift` | Interpretation |
    |--------------|----------------------|---------|
    | Significant | Significant | Changed and moved toward the harmful tail |
    | Significant | Not significant | Changed, but not clearly harmful |
    | Not significant | Not significant | No clear shift |
    | Not significant | Significant | Uncommon — the tail test detects what the broad screen missed |

    Both tests indicate that the risk distribution changed and that the change aligns with higher risk. This is a prompt to investigate, not an automatic decision to retrain. An AUC of `0.5` means chance-level separation; read the harmful-shift statistic against its null distribution and the model's risk scale from 0 to 1. See [How the harm test works](../../explanation/harmful-shift-statistic.md).

=== "Outlier score — confidence — no labels needed"

    `LogitGap` is an **outlier score** for confidence: larger means more certain, so a drop (`worse="lower"`) signals harm. It is the gap between the top logit and the mean of the remaining logits. The risk model fitted in the risk tab supplies the probabilities used here.

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

    There is no evidence of a harmful drop in confidence; the statistic points in the opposite direction, suggesting the model is more certain on the deployment book. With a bad rate near 82%, predicted probabilities tend to polarize toward the extremes and confidence follows. A model can become more confident while also predicting higher risk — confidence complements risk, it does not replace it.

=== "Errors — needs labels"

    Once labels arrive, prediction error (Brier score) offers the clearest post-outcome check. This example uses a random split that is constructed under the null of no error shift, so a large p-value is the expected finding. In a real deployment, a small p-value would be evidence that prediction errors increased in the target.

    ```python
    import numpy as np
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import samesame as ss

    fico = fetch_openml(data_id=45554, as_frame=True)
    X, y = fico.data, fico.target
    y_binary = (y == "Bad").astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.30, stratify=y_binary, random_state=12345,
    )
    rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
    rf.fit(X_train, y_train)
    train_prob = rf.oob_decision_function_[:, 1]
    test_prob = rf.predict_proba(X_test)[:, 1]

    brier_train = (y_train - train_prob) ** 2
    brier_test  = (y_test  - test_prob) ** 2
    harm = ss.test_harmful_shift(source=brier_train, target=brier_test, worse="higher", rng=np.random.default_rng(12345))
    print(f"Brier p={harm.pvalue:.4f}")  # → 0.2737
    ```

    This run finds no evidence that errors worsened. Brier score and log loss usually lead to similar conclusions, so either metric works for this comparison.

To reproduce these examples end to end, see the full runnable scripts in `examples/credit/_code/`.

## Which signal when?

- **Predicted risk** — use when the model output itself represents the harmful outcome.
- **Outlier score — confidence** — use for early warning when labels are not yet available.
- **Prediction error** — use for the clearest post-outcome accuracy check once labels arrive.

Risk and confidence can be monitored without waiting for outcomes, which makes them suitable for early checks. The error comparison waits for labels; in the random-split example nothing is wrong, and the test reflects that. Whether an alarm reflects a change for comparable applicants or the arrival of incomparable ones is the question that [Weight for common support](../weighting/weight-for-common-support.md) addresses.
