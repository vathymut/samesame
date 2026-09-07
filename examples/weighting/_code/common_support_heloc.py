"""Full runnable example for Restrict testing to common support on both sides."""

# --8<-- [start:full]
import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

import samesame as ss

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target
re_obj = re.compile(r"external.*risk.*estimate", flags=re.IGNORECASE)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63
X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)
split = pd.Series([0] * len(X_train) + [1] * len(X_deployment))
X_concat = pd.concat([X_train, X_deployment], ignore_index=True)
domain_probabilities = {}
for label, X_domain in [
    ("all features", X_concat),
    ("excluding split feature", X_concat.drop(columns=[col_split])),
]:
    rf_domain = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=500, random_state=12345, min_samples_leaf=10,
        ),
        method="sigmoid",
        cv=5,
    )
    domain_probabilities[label] = cross_val_predict(
        rf_domain, X_domain, split, cv=5, method="predict_proba",
    )[:, 1]
y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values
rf_bad = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
rf_bad.fit(X_train, y_train_binary)
train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()
unweighted = ss.test_harm(source=train_risk, target=deployment_risk, worse="higher", rng=np.random.default_rng(12345))
print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
for label, domain_prob in domain_probabilities.items():
    source_prob = domain_prob[split.values == 0]
    target_prob = domain_prob[split.values == 1]
    print(f"\n{label}")
    for method, reweight in [
        ("Source-weighted", "source"),
        ("Target-weighted", "target"),
        ("Common-support", "both"),
    ]:
        weights = ss.domain_weights(
            source=source_prob,
            target=target_prob,
            reweight=reweight,
            shrinkage=0.5,
        )
        result = ss.test_harm(
            source=train_risk,
            target=deployment_risk,
            worse="higher",
            weights=weights,
            rng=np.random.default_rng(12345),
        )
        ess = weights.effective_sample_size()
        print(f"{method}: p-value {result.pvalue:.4f}")
        print(f"  ESS source {ess.source:.0f}/{len(source_prob)}, target {ess.target:.0f}/{len(target_prob)}")
# --8<-- [end:full]
