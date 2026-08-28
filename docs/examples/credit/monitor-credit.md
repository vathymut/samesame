# How to: Monitor a credit model

One HELOC split, three signals — pick the one that fits your data.

- **Predicted risk** — no labels needed, directly business-relevant. `worse="higher"`.
- **Confidence** (`LogitGap`) — no labels, monitors certainty. `worse="lower"`.
- **Prediction errors** (Brier / log-loss) — needs labels, most direct accuracy check. `worse="higher"`.

**source** = reference vs **target** = evaluation. Start with [Get started](../tutorials/get-started.md).

## Setup (risk & confidence)

Shared HELOC split — source = higher-risk slice, target = simulated deployment. For errors, see Signal 3 (random split, null is true).

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signal 1: Predicted risk (label-free)

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

    | `test_shift` | `test_harmful_shift` | Meaning |
    |--------------|----------------------|---------|
    | sig. | sig. | changed **and** risk worsened |
    | sig. | not sig. | changed, not clearly harmful |
    | not sig. | sig. | rare — directional where two-sided is not |
    | not sig. | not sig. | no clear harmful shift |

    Both significant here — consider retraining. For AUC `0.5` is chance; for harm compare to null median. See [How the harm test works](../../explanation/harmful-shift-statistic.md).

## Signal 2: Model confidence (label-free)

Use alongside risk when you care about certainty.

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

    Large p-value — confidence shifts *higher* here, so no harmful drop. Risk can rise while confidence also rises (confidently risky). `LogitGap` is an **outlier score** where higher = more certain.

??? details "Signal 3: Prediction errors (needs labels) — when ground truth is available"

    Cleanest signal when you have labels. Random split below (null is true) → expect large `p`.

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

    Large `p` — no evidence model is worse on target. Brier and log-loss usually agree (rank-based); either works. Full scripts: `monitor_prediction_errors_example.py` in `examples/credit/_code/`.

## Next steps

- If overlap is poor, see [Weight for common support](../weighting/weight-for-common-support.md).
- Full scripts: `monitor_credit_risk_example.py`, `monitor_model_confidence_full.py`, `monitor_prediction_errors_example.py` in `examples/credit/_code/`.
