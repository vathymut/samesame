"""Full runnable example for Monitor prediction errors once labels arrive."""

# --8<-- [start:full]
import re

import numpy as np
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
y_deployment = y[~mask_high].reset_index(drop=True)

y_train_binary = y_train.map({"Good": 0, "Bad": 1}).astype(int).values
y_deployment_binary = y_deployment.map({"Good": 0, "Bad": 1}).astype(int).values

rf = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf.fit(X_train, y_train_binary)

train_prob = rf.oob_decision_function_[:, 1]
deployment_prob = rf.predict_proba(X_deployment)[:, 1]

brier_train = (y_train_binary - train_prob) ** 2
brier_deployment = (y_deployment_binary - deployment_prob) ** 2

harm = ss.test_harmful_shift(
    source=brier_train,
    target=brier_deployment,
    worse="higher",
    rng=np.random.default_rng(12345),
)
print(f"Brier p-value: {harm.pvalue:.4f}")
# --8<-- [end:full]
