"""Full runnable example for Monitor prediction errors once labels arrive."""

# --8<-- [start:full]
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

rf = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf.fit(X_train, y_train)

train_prob = rf.oob_decision_function_[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]

brier_train = (y_train - train_prob) ** 2
brier_test = (y_test - test_prob) ** 2

eps = 1e-10
train_clipped = np.clip(train_prob, eps, 1 - eps)
test_clipped = np.clip(test_prob, eps, 1 - eps)
logloss_train = -(y_train * np.log(train_clipped) + (1 - y_train) * np.log(1 - train_clipped))
logloss_test = -(y_test * np.log(test_clipped) + (1 - y_test) * np.log(1 - test_clipped))

for name, s, t in [("Brier", brier_train, brier_test), ("Log-loss", logloss_train, logloss_test)]:
    harm = ss.test_harmful_shift(source=s, target=t, worse="higher", rng=np.random.default_rng(12345))
    print(f"{name} p-value: {harm.pvalue:.4f}")
# --8<-- [end:full]
