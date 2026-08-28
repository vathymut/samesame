"""Full runnable example for Monitor predicted credit risk."""

# --8<-- [start:full]
import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

import samesame as ss

# --- HELOC split: lower-risk vs higher-risk (simulated deployment)
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)

# --- Step 1: domain classifier — is deployment different? (OOB, so honest)
split = pd.Series([0] * len(X_train) + [1] * len(X_deployment))
X_concat = pd.concat([X_train, X_deployment], ignore_index=True)

rf_domain = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf_domain.fit(X_concat, split)
domain_prob = rf_domain.oob_decision_function_[:, 1]

shift = ss.test_shift(
    source=domain_prob[split.values == 0],
    target=domain_prob[split.values == 1],
    rng=np.random.default_rng(12345),
)
print(f"AUC statistic: {shift.statistic:.4f}")
print(f"p-value:       {shift.pvalue:.4f}")
print(
    pd.Series(rf_domain.feature_importances_, index=X_concat.columns)
    .sort_values(ascending=False)
    .head(5)
)

# --- Step 2: credit risk model — did predicted risk rise?
y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values

rf_bad = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)

train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()

harm = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)
print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
# --8<-- [end:full]
