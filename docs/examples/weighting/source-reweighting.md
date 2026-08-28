# How to: Focus harmful-shift testing on common support

Use this when source contains observations deployment will rarely see and you want the test to focus on comparable cases.

This is source-side reweighting: target unchanged, source points unlikely under target are down-weighted.

## Step 1 — Baseline

This example uses the same HELOC split as [Monitor predicted credit risk](../credit/monitor-credit-risk.md).

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-split"
--8<-- "snippets/heloc-split.py:heloc-domain"
--8<-- "snippets/heloc-split.py:heloc-risk-model"

unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",  # or ss.Worse.HIGHER
    rng=np.random.default_rng(12345),
)
```

`domain_prob` (`P(target | x)`) is for weighting only. The harm signal is still predicted default risk — don't reuse `domain_prob` as the harm score.

--8<-- "snippets/honest-scores-ref.txt"

## Step 2 — Build source-side weights

```python
source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]

weights = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",  # or ss.ReweightMode.SOURCE
    shrinkage=0.5,
)

weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights,
    rng=np.random.default_rng(12345),
)

print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
print(f"Weighted   p-value: {weighted.pvalue:.4f}")
```

--8<-- "snippets/clipping-note.txt"

## Step 3 — Read the difference

- **Unweighted** — every observation at full strength.
- **Weighted** — source points unlikely under target are down-weighted (common support from the source side).
- If the result weakens after weighting, the signal was driven by training regions not representative of deployment.
- If it stays strong, the shift persists on common support. Read `.pvalue` for evidence; `.statistic` for magnitude.

Use this mode when training contains outliers or edge cases not representative of the population you now care about.

??? example "Full script — copy and run"
    ```python
    import re

    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier

    import samesame as ss

    # --- HELOC split + domain classifier + risk model (same as credit-risk guide)
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

    # --- Unweighted vs source-weighted harmful shift
    unweighted = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345),
    )
    source_prob = domain_prob[split.values == 0]
    target_prob = domain_prob[split.values == 1]
    weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="source", shrinkage=0.5)
    weighted = ss.test_harmful_shift(
        source=train_risk, target=deployment_risk, worse="higher", weights=weights, rng=np.random.default_rng(12345),
    )
    print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
    print(f"Weighted   p-value: {weighted.pvalue:.4f}")
    print(f"ESS source: {weights.effective_sample_size().source:.1f} / {len(source_prob)}")
    ```

## Next steps

- If deployment also has low-overlap regions, see [Restrict testing to common support on both sides](double-weighting.md).
- Check concentration with [Diagnose weight concentration](diagnose-weight-concentration.md) — worry when `ESS < n/4`.

??? tip "Check weight concentration"
    After building weights, call `weights.effective_sample_size()` to ensure the correction isn't driven by a handful of points.
