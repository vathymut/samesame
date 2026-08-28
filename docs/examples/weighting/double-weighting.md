# How to: Restrict testing to common support on both sides

Use this when both groups have low-overlap regions and source-only reweighting isn't enough. This reweights **both** groups and can substantially change the comparison — use only when low overlap occurs on both sides.

See [Glossary: Reweight](../../explanation/glossary.md#reweight).

## Step 1 — Start from the source-reweighting setup

This guide continues from [Focus harmful-shift testing on common support](source-reweighting.md). At that point you have:

- `source_prob` and `target_prob` — domain probabilities `P(target | x)` from the domain classifier
- `train_risk` and `deployment_risk` — the harm signal (predicted risk)

See that page for the full setup — the code below continues from it.

## Step 2 — Weight both groups

```python
import samesame as ss

weights_both = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="both",  # or ss.ReweightMode.BOTH
    shrinkage=0.5,
)

double_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights_both,
    rng=np.random.default_rng(12345),
)

print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

--8<-- "snippets/clipping-note.txt"

## Step 3 — Compare the three views

Continuing from Step 2 (and [source reweighting](source-reweighting.md) for `unweighted`):

```python
# rebuild unweighted if you didn't run the previous page
unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)

weights_source = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",
    shrinkage=0.5,
)

source_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights_source,
    rng=np.random.default_rng(12345),
)

print(f"Unweighted      p-value: {unweighted.pvalue:.4f}")
print(f"Source-weighted p-value: {source_weighted.pvalue:.4f}")
print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

Three views:

- **Unweighted** — full populations.
- **Source-weighted** — emphasizes overlap from the source side; target unchanged.
- **Doubly-weighted** — emphasizes common support from both sides.

If the signal shrinks only after doubly-weighting, target-side low-overlap points were still influencing the result.

## Choosing `shrinkage` (λ)

--8<-- "snippets/shrinkage-table.txt"

Inspect ESS before lowering — see [Diagnose weight concentration](diagnose-weight-concentration.md). For formulas, see [When importance weights help](../../explanation/importance-weights-rationale.md).

??? example "Full script — copy and run (continues from source-reweighting setup)"
    ```python
    import re

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier

    import samesame as ss

    fico = fetch_openml(data_id=45554, as_frame=True)
    X, y = fico.data, fico.target
    re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
    col_split = next((c for c in X.columns if re_obj.search(c)), None)
    mask_high = X[col_split].astype(float) > 63
    X_train = X[mask_high].reset_index(drop=True)
    y_train = y[mask_high].reset_index(drop=True)
    X_deployment = X[~mask_high].reset_index(drop=True)
    split = pd.Series([0] * len(X_train) + [1] * len(X_deployment))
    X_concat = pd.concat([X_train, X_deployment], ignore_index=True)
    rf_domain = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
    rf_domain.fit(X_concat, split)
    domain_prob = rf_domain.oob_decision_function_[:, 1]
    y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values
    rf_bad = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
    rf_bad.fit(X_train, y_train_binary)
    train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
    deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()
    source_prob = domain_prob[split.values == 0]
    target_prob = domain_prob[split.values == 1]

    unweighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345))
    w_source = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)
    w_both = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
    source_weighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=w_source, rng=np.random.default_rng(12345))
    double_weighted = ss.test_harmful_shift(source=train_risk, target=deployment_risk, worse="higher", weights=w_both, rng=np.random.default_rng(12345))
    print(f"Unweighted      p-value: {unweighted.pvalue:.4f}")
    print(f"Source-weighted p-value: {source_weighted.pvalue:.4f}")
    print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
    for label, w in [("source", w_source), ("both", w_both)]:
        ess = w.effective_sample_size()
        print(f"{label}: ESS source {ess.source:.0f}/{len(source_prob)}, target {ess.target:.0f}/{len(target_prob)}")
    ```

## Next steps

- Diagnose concentration with [Diagnose weight concentration](diagnose-weight-concentration.md):

--8<-- "snippets/ess-rule.txt"

- For the API, see [Importance weights](../../api/weighting.md).
