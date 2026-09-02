"""Full runnable example for Monitor model confidence."""

# --8<-- [start:full]
import re

import numpy as np
from scipy.special import logit
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

import samesame as ss


def logit_gap(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (logits.shape[1] - 1)
    return max_logits - mean_rest

def outlier_scores_from_probabilities(probabilities, *, clip=1e-6):
    clipped = np.clip(np.asarray(probabilities, dtype=float), clip, 1.0 - clip)
    return logit_gap(logit(clipped))

# --- HELOC split (same as credit-risk guide)
fico = fetch_openml(data_id=45554, as_frame=True)
X, y = fico.data, fico.target
re_obj = re.compile(r"external.*risk.*estimate", flags=re.I)
col_split = next((c for c in X.columns if re_obj.search(c)), None)
mask_high = X[col_split].astype(float) > 63
X_train = X[mask_high].reset_index(drop=True)
y_train = y[mask_high].reset_index(drop=True)
X_deployment = X[~mask_high].reset_index(drop=True)

# --- Train credit model (for confidence, not risk)
y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)

# --- Build confidence (outlier) scores — larger = more certain
train_confidence = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
deployment_confidence = outlier_scores_from_probabilities(
    rf_bad.predict_proba(X_deployment)
)
print(f"Training mean confidence:   {train_confidence.mean():.3f}")
print(f"Deployment mean confidence: {deployment_confidence.mean():.3f}")

harm = ss.test_harmful_shift(
    source=train_confidence,
    target=deployment_confidence,
    worse="lower",
    rng=np.random.default_rng(12345),
)
print(f"Statistic: {harm.statistic:.4f}")
print(f"p-value:   {harm.pvalue:.4f}")
# --8<-- [end:full]
