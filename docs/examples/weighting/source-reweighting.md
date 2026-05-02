# How-to: Use source reweighting for adverse-shift testing

**Use this guide when:** you have a model trained on one population and deployed on another
that partially overlaps with training. You want adverse-shift testing to emphasise the shared
region and de-emphasise training samples that are completely foreign to the deployment population.

**What you'll do:**

- Reproduce an unweighted adverse-shift test as a baseline
- Obtain membership probabilities from a separate domain classifier
- Apply `mode="source"` reweighting and compare results

!!! note "Before you start"
    This guide assumes you have completed the tutorial
    [Adjust for covariate shift with importance weights](../tutorials/adjust-for-covariate-shift.md),
    which introduces `contextual_weights` and the `membership_prob` API.

---

## The scenario

You have trained a credit risk model on low-risk customers. The model is now deployed on a
broader population that includes some high-risk customers very unlike anything in training.
You want to test whether predicted default risk shifted adversely, but focus the test on
common support rather than outliers unique to training.

This guide builds on the HELOC dataset setup from
[Monitor a credit risk model](../credit/monitor-credit-risk.md). Complete that guide
first — the data loading and split are identical.

---

## Step 1 — Reproduce the unweighted adverse-shift test

Starting from the HELOC split (training on `ExternalRiskEstimate > 63`, deployment on
`ExternalRiskEstimate <= 63`), build two score streams:

- `membership_prob` from a domain classifier — used for weighting only
- `bad_train` / `bad_test` from a credit model — the adverse-shift scores

```python
import re
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from samesame import test_adverse_shift

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
X_test  = X[~mask_high].reset_index(drop=True)

split = pd.Series([0] * len(X_train) + [1] * len(X_test))
X_concat = pd.concat([X_train, X_test], ignore_index=True)

rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
membership_prob = rf_domain.oob_decision_function_[:, 1]

# Separate harmfulness scores: predicted default risk
loan_status = y[mask_high].reset_index(drop=True).map({"Good": 0, "Bad": 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, loan_status)
bad_train = rf_bad.oob_decision_function_[:, 1].ravel()
bad_test = rf_bad.predict_proba(X_test)[:, 1].ravel()

unweighted = test_adverse_shift(
    source=bad_train,
    target=bad_test,
    direction="higher-is-worse",
    rng=np.random.default_rng(12345),
)
print(f"Unweighted statistic: {unweighted.statistic:.4f}, p-value: {unweighted.pvalue:.4f}")
```

The OOB probabilities from `rf_domain` are out-of-sample estimates of `P(deployment | x)`
and go directly into `membership_prob`. They are never used as adverse-shift scores.

---

## Step 2 — Apply source reweighting

Pass `membership_prob` and `mode="source"` to `test_adverse_shift`. Source samples that look
unlike any deployment sample receive lower weights, so the adverse-shift test focuses on overlap:

```python
weighted = test_adverse_shift(
    source=bad_train,
    target=bad_test,
    direction="higher-is-worse",
    membership_prob=membership_prob,
    mode="source",
    alpha_blend=0.5,
    rng=np.random.default_rng(12345),
)
print(f"Weighted   statistic: {weighted.statistic:.4f}, p-value: {weighted.pvalue:.4f}")
```

---

## Step 3 — Compare weighted vs unweighted results

| Test | Interpretation |
|------|----------------|
| Unweighted | Harm signal across both populations, including source-only outliers. |
| Source-reweighted | Harm signal restricted to common support; source outliers down-weighted. |

If unweighted is significant but weighted is not, adverse shift may be concentrated in
low-overlap source regions. If both are significant, the adverse shift persists in common support.

---

## When to use source reweighting

- Common support between training and deployment is narrow.
- Training contains many samples with feature values never seen in deployment.
- You want adverse-shift testing to focus on the subpopulation the model actually encounters.

---

## See also

- [Use double-weighting for covariate-shift adaptation](double-weighting.md)
  — when deployment also contains outliers foreign to training.
- [Why importance weights stabilise shift detection](../../explanation/importance-weights-rationale.md)
  — conceptual background on RIW and `alpha_blend`.
- [Weighting strategies](../../api/weighting.md) — full API reference for `membership_prob` and `mode`.
