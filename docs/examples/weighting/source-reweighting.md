# How to: Focus harmful-shift testing on shared support

Use this guide when the source group contains observations that deployment will rarely or never see,
and you want the harmful-shift test to focus on comparable cases instead.

This is the practical version of source reweighting: keep deployment unchanged, and down-weight the
source observations that are unlikely under the target distribution.

## Step 1 - Recreate the baseline workflow

This example uses the same HELOC split as
[Monitor predicted credit risk](../credit/monitor-credit-risk.md).

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
X_deployment = X[~mask_high].reset_index(drop=True)

split = pd.Series([0] * len(X_train) + [1] * len(X_deployment))
X_concat = pd.concat([X_train, X_deployment], ignore_index=True)

rf_domain = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
domain_prob = rf_domain.oob_decision_function_[:, 1]

y_train_binary = y[mask_high].reset_index(drop=True).map({"Good": 0, "Bad": 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)

train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()

unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)
```

`domain_prob` is for weighting only. The harmful-shift signal is still predicted default risk.

## Step 2 - Build source-side importance weights

```python
from samesame.weights import domain_weights

source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]

weights = domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",
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

## Step 3 - Interpret the difference

- The unweighted test uses every observation at full strength.
- The weighted test reduces the influence of source observations that are unlikely under the target
  distribution.
- If the result weakens substantially, the original signal was being driven by parts of training
  that are not very relevant for deployment.
- If the result stays strong, the shift persists in the region where the two groups have common
  support.

Use this mode when training contains outliers or edge cases that are not representative of the
population you now care about.

If deployment also contains many low-overlap cases, continue to
[Restrict testing to common support on both sides](double-weighting.md).
