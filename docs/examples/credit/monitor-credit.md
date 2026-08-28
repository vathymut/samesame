# How to: Monitor a credit model

One HELOC split, three signals. Pick the one that fits your data.

- **Predicted risk** — no labels needed, directly business-relevant. `worse="higher"`.
- **Confidence** (`LogitGap`) — no labels, monitors certainty. `worse="lower"`.
- **Prediction errors** (Brier / log-loss) — needs labels, most direct accuracy check. `worse="higher"`.

All use **source** = reference vs **target** = evaluation. See [Glossary](../../explanation/glossary.md).

!!! info "Prerequisites"
    - [Detect any shift](../tutorials/detect-distribution-shift.md) — shift test and honest scores.
    - [Is it harmful?](../tutorials/check-shift-harm.md) — `worse` and direction.

## Setup (risk & confidence)

Shared HELOC split — source = higher-risk slice, target = simulated deployment. For prediction errors, see Signal 3 below (random split, null is true).

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

--8<-- "snippets/honest-scores-ref.txt"

---

## Signal 1: Predicted risk (label-free, business impact)

Best when your model output already has business meaning.

=== "Code"

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
    print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")   # → 0.9999, 0.0001
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")    # → ~0.35, 0.0001
    ```

=== "How to read"

    --8<-- "snippets/pvalue-guidance.txt"

    | `test_shift` | `test_harmful_shift` | Meaning |
    |--------------|----------------------|---------|
    | sig. | sig. | changed **and** risk worsened |
    | sig. | not sig. | changed, not clearly harmful |
    | not sig. | sig. | rare — directional signal where two-sided is not |
    | not sig. | not sig. | no clear harmful shift |

    On this split both are significant — consider retraining or policy change.

??? tip "Why OOB?"
    `oob_decision_function_` is free for `RandomForestClassifier(oob_score=True)` and honest by construction. For other estimators use `cross_val_predict(cv=10, method="predict_proba")`.

## Signal 2: Model confidence (label-free, certainty)

Use alongside risk when you care about certainty, not just the predicted probability.

=== "Code"

    ```python
    from sklearn.ensemble import RandomForestClassifier
    import samesame as ss

    bad_mapping = {"Good": 0, "Bad": 1}
    y_train_binary = y_train.map(bad_mapping).values
    rf_bad = RandomForestClassifier(
        n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
    )
    rf_bad.fit(X_train, y_train_binary)

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
    print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")  # → ~0.04, ~0.90
    ```

=== "How to read"

    Large p-value here — confidence shifts *higher* on this split, so no harmful drop. This doesn't contradict Signal 1: risk can rise while confidence also rises (confidently risky).

Confidence is an **outlier score** where higher = more certain. See [Glossary: Score](../../explanation/glossary.md#score).

## Signal 3: Prediction errors (once labels arrive)

Cleanest signal when you have ground truth.

=== "Setup — random split (null is true)"

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
    ```

=== "Brier & log-loss"

    ```python
    brier_train = (y_train - train_prob) ** 2
    brier_test  = (y_test  - test_prob) ** 2

    eps = 1e-10
    train_clipped = np.clip(train_prob, eps, 1 - eps)
    test_clipped  = np.clip(test_prob, eps, 1 - eps)
    logloss_train = -(y_train * np.log(train_clipped) + (1 - y_train) * np.log(1 - train_clipped))
    logloss_test  = -(y_test  * np.log(test_clipped)  + (1 - y_test)  * np.log(1 - test_clipped))

    harm_brier = ss.test_harmful_shift(source=brier_train, target=brier_test, worse="higher", rng=np.random.default_rng(12345))
    harm_logloss = ss.test_harmful_shift(source=logloss_train, target=logloss_test, worse="higher", rng=np.random.default_rng(12345))
    print(f"Brier p={harm_brier.pvalue:.4f} Log-loss p={harm_logloss.pvalue:.4f}")  # → ~0.60, ~0.60
    ```

    Large p-values — random split, no evidence the model is worse on target. Brier and log-loss usually agree (rank-based test).

## Next steps

- If overlap is poor, see [Weight for common support](../weighting/weight-for-common-support.md).
- Full runnable scripts: `monitor_credit_risk_example.py`, `monitor_model_confidence_full.py`, `monitor_prediction_errors_example.py` in `docs/examples/credit/_code/`.
