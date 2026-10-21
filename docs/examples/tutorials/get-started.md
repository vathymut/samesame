# Get started

This tutorial shows how to turn scores into two answers: *did it change?* and *did it get worse?*

## Prerequisites

- Python 3.12+ with `numpy`, `scipy`, `scikit-learn`, and `samesame` installed.
- Basic familiarity with p-values and classifiers. You will use `cross_val_predict` once.

--8<-- "snippets/source-target.txt"

## Steps

Use one score per row. `test_shift` asks whether source and target score
distributions differ at all. `test_harm` tests whether target scores move
toward a harmful tail you declare with `worse` before testing.

### 1. Create source and target

Draw a reference sample (`source`) and a sample from the population you are checking (`target`). The reference is usually training data or a past deployment.

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

The first feature differs between the groups; the other three come from the same distribution. The classifier has a clear shift to find without extra differences in those features.

### 2. Score out of sample

Score each row out of sample: the model must not have been fit on that row.
In-sample predictions can exaggerate source-target separation.

Here, a domain classifier estimates `P(target|x)`, the probability that a row
belongs to target. If it can distinguish source from target, its probabilities
give a general signal of shift. For background on classifier-based two-sample
tests, see [In gentle praise of classifier tests](https://vathymut.org/posts/2022-01-22-in-gentle-praise-of-modern-tests/).

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

    Pass `rng=np.random.default_rng(12345)` for reproducible p-values. The default is `n_resamples=9999`; use `999` while exploring and `19999` for finer resolution below `0.001`. Details: [Shift testing](../../api/testing.md).

### 3. Did anything change?

Now test whether the domain score separates source from target. Because we introduced a difference in the first feature, you should expect clear separation here:

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"Shift p-value: {shift.pvalue:.4f}")  # → 0.0002
```

A small p-value is evidence against the null of no shift. Here, that is the difference we built into the first feature.

### 4. Did it get worse?

The domain probability tells how likely a row is to belong to target. Use it
in `test_harm` only if target-like observations are harmful in your setting.
Because higher values mean more target-like, use `worse="higher"` for that
case.

```python
harm = ss.test_harm(
    source=source_scores,
    target=target_scores,
    worse="higher",
    rng=rng,
)
print(f"Harm p-value: {harm.pvalue:.4f}")  # → 0.0002
```

A small p-value is evidence against the null of no harmful shift. Here,
target scores appear more often in the high tail. That describes group
membership, not outcome harm; the interpretation depends on the direction
you declared.

### 5. Use a score tied to your outcome

The domain probability in this demo measures group membership, not outcome
quality. Use it as a harm score only if target-like observations are harmful
in your setting. For the outcome you care about, choose a score tied to it
and declare `worse` from its meaning before calling `test_harm`. Higher risk
or error is worse; lower confidence or quality is worse. See [Shift testing](../../api/testing.md)
for function signatures and direction values.

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
    rng = np.random.default_rng(12345)
    source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
    target_quality = rng.normal(loc=0.72, scale=0.07, size=400)  # shift down: harmful
    harm = ss.test_harm(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

## Recap

Use `test_shift` to ask whether the score distributions differ. Use `test_harm` to ask whether target moved toward the harmful tail you named before testing. A detectable shift is not, by itself, a harmful shift.

Next:

- [Is the new drug good enough?](../trials/check-drug-efficacy.md): the same test on 70 trial scores, with no model.
- [Monitor a credit model](../credit/monitor-credit.md): risk, confidence, and error on a HELOC book.
- [How the harm test works](../../explanation/harmful-shift-statistic.md): why the weighted AUC leans on the harmful tail.
