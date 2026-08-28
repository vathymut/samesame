# Get started

This tutorial builds the smallest useful monitoring workflow. You will turn
observations into one score each, test whether source and target differ, and
then test a declared harmful direction.

Use **source** for the reference distribution (for example, training data or a
past deployment) and **target** for the current deployment. The score can be
predicted risk, prediction error, confidence, or an outlier score.

- `ss.test_shift` — did anything change? Two-sided AUC. `0.5` is chance.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did target move into the harmful tail you declare? One-sided. Larger always means worse (`worse="lower"` flips the sign).

`test_shift` is a broad, two-sided screen. `test_harmful_shift` is a focused,
one-sided test: it asks whether target puts more mass beyond thresholds that
few source observations exceed, after orienting the score so that larger means
worse. The same overall shift can therefore be harmless, harmful, or even
beneficial depending on where it occurs. See [How the harm test works](../../explanation/harmful-shift-statistic.md) for the intuition and formula.

Read `.pvalue` as evidence against the relevant null, then inspect the
statistic and the distributions. A small p-value says the observed pattern is
unlikely under label exchangeability; it does not say the change is large or
caused by deployment.

## 1 — Make source and target

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, source might be training rows and target deployment rows. The
choice should match the operational question: every conclusion is relative to
the source reference you selected.

## 2 — Score out of sample

Each row must be scored by a model that did not see that row. Here a domain
classifier estimates `P(target|x)`, the probability that a row looks like it
came from target. This is a useful generic score for detecting *any* shift:

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

- Small `test_shift` p - the score distributions differ.
- Small `test_harmful_shift` p - target also shifted toward the tail you
  declared.

Flip `worse` and the conclusion can disappear: the test is directional by
design. Choose `worse` from the meaning of the score before looking at the
result, rather than choosing whichever direction gives a smaller p-value.

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` for deterministic p-values. Default `n_resamples=9999` (`999` while exploring, `19999` for `p < 0.001`).

When feature support differs, see [Weight for common support](../weighting/weight-for-common-support.md).
