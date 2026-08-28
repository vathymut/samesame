# --8<-- [start:heloc-split]
import re

import pandas as pd
from sklearn.datasets import fetch_openml

fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target

re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63

X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)

print(f"Training set:   {len(X_train)} samples")
print(f"Deployment set: {len(X_deployment)} samples")
# --8<-- [end:heloc-split]

# --8<-- [start:heloc-domain]
import pandas as pd  # kept for snippet self-containment when included alone
from sklearn.ensemble import RandomForestClassifier

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
# --8<-- [end:heloc-domain]

# --8<-- [start:heloc-risk-model]
from sklearn.ensemble import RandomForestClassifier

y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values

rf_bad = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=12345,
    min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)

train_risk = rf_bad.oob_decision_function_[:, 1].ravel()
deployment_risk = rf_bad.predict_proba(X_deployment)[:, 1].ravel()
# --8<-- [end:heloc-risk-model]
