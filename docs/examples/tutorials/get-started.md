# Get started

This tutorial shows how to turn scores into two answers: *did it change?* and *did it get worse?*

## Prerequisites

- Python 3.12+ with `numpy`, `scipy`, `scikit-learn`, and `samesame` installed.
- Comfort with p-values and training a classifier (you will use `cross_val_predict` once).

--8<-- "snippets/source-target.txt"

## Steps

Run both tests on the score for the outcome you care about. `test_shift` asks whether source and target differ at all. `test_harm` asks whether target moved toward the harmful tail you name with `worse`.

### 1. Create source and target

Draw a reference sample and a sample from the population you are checking. The reference is usually training data or a past deployment.

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

The target differs from the source in its first feature. That gives the classifier a deliberate shift to find, while the remaining features stay aligned.

### 2. Score out of sample

Each row must be scored by a model that did not see that row. Here, a domain classifier estimates `P(target|x)`, the probability that an observation looks like it came from the target population. This is a useful generic score for detecting *any* shift because it measures sample membership. For background on classifier-based two-sample tests, see [In gentle praise of classifier tests](https://vathymut.org/posts/2022-01-22-in-gentle-praise-of-modern-tests/).

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

??? note "Honest and reproducible scores"
    `samesame` only sees the scores you pass in.

    --8<-- "snippets/honest-scores.txt"

    Pass `rng=np.random.default_rng(12345)` for reproducible p-values (`n_resamples=9999`; use `999` while exploring and `19999` below `0.001`). Details: [Shift testing](../../api/testing.md).

### 3. Did anything change?

Now test whether the domain score separates source from target. Because we introduced a difference in the first feature, you should expect clear separation here:

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"Shift p-value: {shift.pvalue:.4f}")  # → 0.0002
```

The p-value is small, so we reject no shift. That is the difference built into the first feature.

### 4. Did it get worse?

The domain probability can also serve as an outlier score when target-like observations represent the harmful direction: a higher domain probability means the observation looks less like the reference sample, so use `worse="higher"` when these deviations are harmful.

```python
harm = ss.test_harm(
    source=source_scores,
    target=target_scores,
    worse="higher",
    rng=rng,
)
print(f"Harm p-value: {harm.pvalue:.4f}")  # → 0.0002
```

We reject the null of no harmful shift. Target scores sit further in the high tail you marked harmful: rows that look like the target, not the reference. That is a membership result. It counts as harm only because you declared target-like scores harmful.

### 5. Declare the harmful direction

Domain probability is only one score. Test any score tied to the outcome you care about. Higher is harmful for risk and error. Lower is harmful for confidence and quality. Replace the draws below with your own score. Signatures and polarity: [Shift testing](../../api/testing.md).

--8<-- "snippets/worse-declaration.txt"

--8<-- "snippets/worse-table.txt"

=== "Higher is worse (risk, error)"

    ```python
    rng = np.random.default_rng(12345)
    source_risk = rng.normal(loc=0.20, scale=0.07, size=400)
    target_risk = rng.normal(loc=0.28, scale=0.07, size=400)  # shift up: harmful
    harm = ss.test_harm(source=source_risk, target=target_risk, worse="higher", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

=== "Lower is worse (confidence)"

    ```python
    source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
    target_quality = rng.normal(loc=0.72, scale=0.07, size=400)  # shift down: harmful
    harm = ss.test_harm(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

## Recap

You have one score and two tests. `test_shift` says whether source and target differ. `test_harm` says whether target moved toward the harmful tail you named before testing. A detectable shift is not, by itself, a harmful shift.

Next:

- [Is the new drug good enough?](../trials/check-drug-efficacy.md): the same test on 70 trial scores, with no model.
- [Monitor a credit model](../credit/monitor-credit.md): risk, confidence, and error on a HELOC book.
- [How the harm test works](../../explanation/harmful-shift-statistic.md): why the weighted AUC leans on the harmful tail.
