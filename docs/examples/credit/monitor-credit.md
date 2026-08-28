# Monitor a credit model

One HELOC split, three interchangeable signals. Pick what fits your data:

| Signal | Needs labels? | Harm is… | `worse` |
|--------|---------------|----------|---------|
| Predicted risk | No | higher risk | `higher` |
| Confidence (`LogitGap`) | No | lower certainty | `lower` |
| Prediction error (Brier) | Yes | larger error | `higher` |

Start with risk if your model output already means harm. Use confidence when you care about certainty; use errors when labels have arrived.

**Source** = reference vs **target** = evaluation. New here? Start with [Get started](../tutorials/get-started.md).

## Setup

Same HELOC split for risk and confidence — source = higher-risk slice, target = simulated deployment. Errors use a separate random split where the null is true.

```python
--8<-- "snippets/heloc-split.py:heloc-split"
```

## Signal 1 — Predicted risk (no labels needed)

Directly business-relevant.

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-domain"
--8<-- "snippets/heloc-split.py:heloc-risk-model"

shift = ss.test_shift(
    source=domain_prob[split.values == 0],
    target=domain_prob[split.values == 1],
    rng=np.random.default_rng(12345),
)
harm = ss.test_harmful_shift(
    source=train_risk, target=deployment_risk,
    worse="higher", rng=np.random.default_rng(12345),
)
print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")   # → 0.9999, 0.0001
print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")    # → 0.35, 0.0001
```

Reading both together:

| `test_shift` | `test_harmful_shift` | Meaning |
|--------------|----------------------|---------|
| sig. | sig. | changed **and** moved toward worse |
| sig. | not sig. | changed, not clearly harmful |
| not sig. | not sig. | no clear shift |

Both significant here — consider retraining. AUC `0.5` is chance; for harm compare observed to null median (see [How the harm test works](../../explanation/harmful-shift-statistic.md)).

## Signal 2 — Model confidence (no labels needed)

An **outlier score** where larger = more certain. `LogitGap` = gap between the top logit and the mean of the rest — confident predictions stand out.

```python
from sklearn.ensemble import RandomForestClassifier
import samesame as ss

y_train_binary = y_train.map({"Good": 0, "Bad": 1}).values
rf_bad = RandomForestClassifier(
    n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10,
)
rf_bad.fit(X_train, y_train_binary)

--8<-- "examples/credit/_code/monitor_model_confidence_example.py:imports"
--8<-- "examples/credit/_code/monitor_model_confidence_example.py:logit-gap"
--8<-- "examples/credit/_code/monitor_model_confidence_example.py:outlier-scores"

train_conf = outlier_scores_from_probabilities(rf_bad.oob_decision_function_)
deploy_conf = outlier_scores_from_probabilities(rf_bad.predict_proba(X_deployment))

harm = ss.test_harmful_shift(
    source=train_conf, target=deploy_conf,
    worse="lower",  # lower confidence = harm
    rng=np.random.default_rng(12345),
)
print(f"Harm {harm.statistic:.4f} p={harm.pvalue:.4f}")  # → 0.04, 0.90
```

Large `p` — confidence actually rose, so no harmful drop. Risk can rise while confidence also rises (confidently risky).

## Signal 3 — Prediction errors (needs labels)

Most direct accuracy check when ground truth is available. Random split below (null is true) → expect large `p`.

```python
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
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=12345, min_samples_leaf=10)
rf.fit(X_train, y_train)
train_prob = rf.oob_decision_function_[:, 1]
test_prob = rf.predict_proba(X_test)[:, 1]

brier_train = (y_train - train_prob) ** 2
brier_test  = (y_test  - test_prob) ** 2
harm = ss.test_harmful_shift(source=brier_train, target=brier_test, worse="higher", rng=np.random.default_rng(12345))
print(f"Brier p={harm.pvalue:.4f}")  # → ~0.60
```

Large `p` — no evidence errors got worse on target. Brier and log-loss usually agree here; either works.

## Which signal when?

- **Risk** — default choice, closest to business harm.
- **Confidence** — early warning before labels arrive; use alongside risk.
- **Errors** — cleanest once labels arrive, but delayed.

## Next steps

- Poor overlap? See [Weight for common support](../weighting/weight-for-common-support.md).
- Full runnable scripts: `examples/credit/_code/` (`monitor_credit_risk_example.py`, `monitor_model_confidence_full.py`, `monitor_prediction_errors_example.py`).
