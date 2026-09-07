# Get started

One score, two tests. You'll ask *did it change?* and *did it get worse?* about a score of your own, and you'll leave with a habit you can reuse.

## Prerequisites

- Python 3.12+ with `numpy`, `scikit-learn`, and `samesame` installed.
- Comfort with p-values and training a classifier (you will use `cross_val_predict` once).

--8<-- "snippets/source-target.txt"

## Steps

You'll use two tests for two different questions. `ss.test_shift` is broad and two-sided (AUC `0.5` means no separation). `ss.test_harmful_shift(..., worse="higher"|"lower")` is focused and one-sided on the tail you name in advance.

### 1. Create source and target

Create the two populations your question compares, for example training versus current deployment. Every conclusion you'll draw describes target relative to source.

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

### 2. Score out of sample

Score each row with a model that didn't see it. Here a domain classifier estimates the domain probability `P(target|x)`, a handy generic score for catching *any* shift. It measures membership, not outcome quality, so keep it separate from the score you'll judge for harm.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

### 3. Did anything change?

Test whether the score separates source from target (expect clear separation here):

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"Shift p-value: {shift.pvalue:.4f}")  # → 0.0002
```

Near `0.5` means little separation; values near `0` or `1` mean stronger separation. The test is two-sided, so a shift in either direction can reject.

### 4. Did it get worse?

Pick the harmful direction from what the score means, *before* you look at results. Never choose `worse` by p-value. See [Core concepts](../../explanation/core-concepts.md) for the full table.

--8<-- "snippets/worse-declaration.txt"

--8<-- "snippets/worse-table.txt"

=== "Higher is worse (risk, error)"

    ```python
    rng = np.random.default_rng(12345)
    source_risk = rng.normal(loc=0.20, scale=0.07, size=400)
    target_risk = rng.normal(loc=0.28, scale=0.07, size=400)  # shift up: harmful
    harm = ss.test_harmful_shift(source=source_risk, target=target_risk, worse="higher", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

=== "Lower is worse (confidence)"

    ```python
    source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
    target_quality = rng.normal(loc=0.72, scale=0.07, size=400)  # shift down: harmful
    harm = ss.test_harmful_shift(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

- A small `test_shift` p-value means the distributions differ.
- A small `test_harmful_shift` p-value means target moved toward the tail you named.

??? note "Honest and reproducible scores"
    `samesame` only sees the scores you pass in. --8<-- "snippets/honest-scores.txt"

    Pass `rng=np.random.default_rng(12345)` for reproducible p-values (`n_resamples=9999`; use `999` while exploring and `19999` below `0.001`). Details: [Core concepts](../../explanation/core-concepts.md).

## Recap

One score, two verdicts. `test_shift` screens for any difference between the groups; `test_harmful_shift` asks whether target moved toward the tail you fixed in advance.

Where you'd like to go next:

- [Is the new drug good enough?](../trials/check-drug-efficacy.md): the same test on 70 trial scores, with no model to fit.
- [Weight for common support](../../how-to/weight-for-common-support.md): when overlap is poor, reweight around common support.
- [How the harm test works](../../explanation/harmful-shift-statistic.md): why the weighted AUC leans into the harmful tail.
