# Get started

One score per observation — predicted risk, prediction error, or outlier score — for **source** (reference: training or past deployment) vs **target** (current deployment).

- `ss.test_shift` — did anything change? Two-sided AUC. `0.5` is chance.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did target move into the harmful tail you declare? One-sided. Larger always means worse (`worse="lower"` flips the sign).

The difference is weighting: AUC averages uniformly (`∫ TPR dFPR`); harm emphasizes low `FPR` (`∫ TPR·(1−FPR)² dFPR`) — thresholds few source points clear but many target points do. Same AUC can hide opposite harm; see [How the harm test works](../../explanation/harmful-shift-statistic.md).

Read `.pvalue` first (≤ 0.05 is evidence against the null), `.statistic` second.

## 1 — Make source and target

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, source = training rows and target = deployment rows.

## 2 — Score out of sample

Each row must be scored by a model that didn't see it. Here a domain classifier gives `P(target|x)` — a valid score for detecting *any* shift:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

!!! warning "Honest scores"
    `samesame` only sees scores — not how they were made. If scores come from a fitted model, generate them out of sample (`cross_val_predict`, `oob_decision_function_`, or held-out set). In-sample predictions create false separation and invalidate the test.

For harm tests on your own business score (risk, error, confidence), keep the domain probability separate — use it to build weights, not as the harm score.

## 3 — Did anything change?

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"AUC {shift.statistic:.3f} p={shift.pvalue:.4f}")  # → 0.611, 0.0002
```

Farther from `0.5` = stronger separation. Two-sided, so both `0.8` and `0.2` reject.

## 4 — Did it get worse?

Declare the harmful direction once — string or `ss.Worse` (interchangeable):

--8<-- "snippets/worse-table.txt"

=== "Higher is worse (risk, error)"

    ```python
    rng = np.random.default_rng(12345)
    source_risk = rng.normal(loc=0.20, scale=0.07, size=400)
    target_risk = rng.normal(loc=0.28, scale=0.07, size=400)  # shift up — harmful
    harm = ss.test_harmful_shift(source=source_risk, target=target_risk, worse="higher", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

=== "Lower is worse (confidence)"

    ```python
    source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
    target_quality = rng.normal(loc=0.72, scale=0.07, size=400)  # shift down — harmful
    harm = ss.test_harmful_shift(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

- Small `test_shift` p — groups differ.
- Small `test_harmful_shift` p — target also shifted toward the tail you declared.

Flip `worse` and p goes to ~1 — the test is directional.

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` for deterministic p-values. Default `n_resamples=9999` (`999` while exploring, `19999` for `p < 0.001`).

When feature support differs, see [Weight for common support](../weighting/weight-for-common-support.md).
