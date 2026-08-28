# Tutorial: Detect any distributional shift

Full loop in 5 minutes: build a score, keep it honest, run `ss.test_shift`.

**Source** = reference (training). **Target** = evaluation (production). Idea: train a classifier to distinguish them. If its out-of-sample `P(target|x)` separates the groups better than chance, they differ.

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

Each row must be scored by a model that didn't train on it.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

`domain_prob` is `P(target|x)` — the signal for `test_shift` only.

--8<-- "snippets/honest-scores.txt"

## 3 — Run the shift test

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"AUC {shift.statistic:.3f} p={shift.pvalue:.4f}")  # → 0.611, 0.0002
```

AUC `0.5` = chance; farther from `0.5` means stronger separation. `test_shift` is two-sided, so `0.2` also rejects — ordering is reversed.

--8<-- "snippets/pvalue-guidance.txt"

This answers only "did anything change?" For harm, see [Is it harmful?](check-shift-harm.md).

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)`. --8<-- "snippets/n-resamples.txt"
