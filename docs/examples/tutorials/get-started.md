# Get started

A single score per observation — e.g., predicted risk, prediction error, or outlier score — for **source** (reference) vs **target** (evaluation). Two questions, two functions.

- `ss.test_shift` — did anything change? Two-sided.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did target move toward worse outcomes? One-sided, needs a direction.

5 minutes with synthetic data. Then swap in your own scores.

## 1 — Make source and target

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, source = training rows, target = production rows.

## 2 — Score out of sample

Each row must be scored by a model that didn't see it.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

`domain_prob` is `P(target|x)`. Use it as the score for `test_shift` only.

--8<-- "snippets/honest-scores.txt"

## 3 — Did anything change?

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"AUC {shift.statistic:.3f} p={shift.pvalue:.4f}")  # → 0.611, 0.0002
```

AUC `0.5` = chance; farther from `0.5` means stronger separation. Two-sided, so `0.2` also rejects when ordering is reversed.

--8<-- "snippets/pvalue-guidance.txt"

This answers only "did anything change?" For direction, continue.

## 4 — Did it get worse?

Declare which direction is harmful — string or `ss.Worse` (interchangeable):

--8<-- "snippets/worse-table.txt"

Internally scores are negated when `worse="lower"` so larger always means worse.

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
    shift = ss.test_shift(source=source_quality, target=target_quality, rng=rng)
    harm = ss.test_harmful_shift(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Shift p={shift.pvalue:.4f} Harm p={harm.pvalue:.4f}")  # → 0.0001, 0.0001
    ```

How to read:

- Small `test_shift` p — groups differ.
- Small `test_harmful_shift` p — target also shifted toward the harmful tail you declared.

When feature support differs, see [Weight for common support](../weighting/weight-for-common-support.md). For the statistic, see [How the harm test works](../../explanation/harmful-shift-statistic.md).

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)`. --8<-- "snippets/n-resamples.txt"
