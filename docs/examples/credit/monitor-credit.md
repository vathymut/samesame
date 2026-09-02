# Monitor a credit model

An alarm should tell you whether the score moved toward worse outcomes, not
just that it moved. Using one HELOC split (External Risk Estimate >63 vs
≤63, Gardner et al. 2023), this example compares three scores from the same
model — they answer different questions and arrive at different times.

| Signal | Requires labels? | Harmful direction | `worse` |
|--------|-------------------|-------------------|---------|
| Predicted risk | No | Higher risk | `higher` |
| Confidence (`LogitGap`) | No | Lower certainty | `lower` |
| Prediction error (Brier) | Yes | Larger error | `higher` |

Start with predicted risk when the model output already represents a harmful
outcome. Use confidence for an early warning when labels are delayed, and
prediction error for the clearest post-outcome check. **Source** is the
reference distribution, such as training data or a past deployment; **target**
is the current deployment. If you are new to `samesame`, start with [Get
started](../tutorials/get-started.md).

## Setup

The same HELOC split (source: training lower-risk, target: deployment
higher-risk) is reused for risk and confidence. The error tab uses a separate
random train/test split where the null of no error shift holds, so a large
p-value is expected.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signals

=== "Risk — no labels needed"

    Predicted risk — model's `P(default)`; larger = worse, so `worse="higher"`
    (pre-specify, don't pick by p-value, cf. `Worse`). Higher values represent
    a worse outcome.

    ```python
    import numpy as np
    import samesame as ss

    --8<-- "snippets/heloc-split.py:heloc-domain"
    --8<-- "snippets/heloc-split.py:heloc-risk-model"

    # any shift? domain_prob is generic; harm on interpretable risk
    shift = ss.test_shift(
        source=domain_prob[split.values == 0],
        target=domain_prob[split.values == 1],
        rng=np.random.default_rng(12345),
    )
    harm = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk,
        worse="higher", rng=np.random.default_rng(12345),
    )
    print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")   # → 0.9999, 0.0001
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")    # → 0.35, 0.0001
    ```

    | `test_shift` | `test_harmful_shift` | Meaning |
    |--------------|----------------------|---------|
    | sig. | sig. | changed **and** moved toward worse |
    | sig. | not sig. | changed, not clearly harmful |
    | not sig. | not sig. | no clear shift |
    | not sig. | sig. | rare — directional where broad is not |

    Both tests provide evidence that the risk distribution changed and that the
    change is consistent with higher risk. This is a reason to investigate, not an
    automatic retraining decision. An AUC of `0.5` is chance; read the
    harmful-shift statistic against its null distribution and the model's risk
    scale (0–1). See [How it works](../../explanation/harmful-shift-statistic.md).

=== "Confidence — no labels needed"

    `LogitGap` is an **outlier score** for confidence: larger = more certain,
    so a drop (`worse="lower"`) is harm. It is the gap between the top logit
    and the mean of the remaining logits.

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
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")  # → 0.04, 0.90
    ```

    The large `p`-value provides no evidence of a harmful confidence drop in this test, but it does not prove that confidence is unchanged. Confidence and risk can move together: a model may become more confident while also becoming more risky. This example assumes `X_train`, `X_deployment`, and `rf_bad` from Setup.

=== "Errors — needs labels"

    Once labels arrive, prediction error (Brier) is the clearest post-outcome
    check. The random split here is deliberately constructed under the null of
    no error shift, so a large p-value is expected. In a real deployment, a
    small p-value would provide evidence that prediction errors increased in
    the target.

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
    print(f"Brier p={harm.pvalue:.4f}")  # → ~0.60
    ```

    This example provides no evidence that errors got worse. Brier score and log loss usually lead to similar conclusions, so either metric can be used for this comparison.

To reproduce these examples end to end, see the full runnable scripts in `examples/credit/_code/`.

## Which signal when?

- **Predicted risk** — use when the model output already represents the harmful outcome.
- **Confidence** — use for an early warning when labels are not yet available.
- **Prediction error** — use for the clearest post-outcome accuracy check.

If source and target have poor overlap in feature space, see [Weight for common support](../weighting/weight-for-common-support.md).
