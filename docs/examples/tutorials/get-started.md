# Get started

This tutorial uses one score per observation to ask two questions: *do the
groups separate by score?* and *does target move toward the tail you marked
harmful?*

## Prerequisites

- Python 3.12+ with `numpy`, `scipy`, `scikit-learn`, and `samesame` installed.
- Basic familiarity with p-values. You do not need to know classifiers; the
  tutorial explains why each row must be scored out of sample.

--8<-- "snippets/source-target.txt"

## Steps

### 1. Create source and target

Draw a reference sample (`source`) and a sample from the population you are
checking (`target`). The reference is usually training data or a past
deployment.

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

The first feature shifts between the groups; the other three do not. This
gives the classifier one clear signal to detect.

### 2. Score out of sample

Score each row out of sample: the model must not have been fit on that row.
In-sample predictions can exaggerate source-target separation.

Here, a domain classifier estimates `P(target|x)`, the probability that a row
belongs to target. Its probabilities show how target-like each row appears.
For background on classifier-based two-sample tests, see
[In gentle praise of classifier tests](https://vathymut.org/posts/2022-01-22-in-gentle-praise-of-modern-tests/).

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

    Pass `rng=np.random.default_rng(12345)` for reproducible p-values. The
    default is `n_resamples=9999`; use `999` while exploring and `19999` for
    finer p-value resolution, especially below `0.001`. See
    [Shift testing](../../api/testing.md) for details.

### 3. Do the scores separate?

Test whether the domain score separates source from target. Because we
introduced a difference in the first feature, expect clear separation:

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"Shift p-value: {shift.pvalue:.4f}")  # → 0.0002
```

A small p-value means the observed separation would be unusual if source and
target labels were unrelated to the pooled scores. Here, it reflects the
first-feature difference we built in.

### 4. Did it get worse?

Here, the domain probability measures group membership, not outcome quality:
higher values mean a row looks more like target. Use `worse="higher"` only if
target-like observations are harmful in your setting. This example shows how
the test works; domain probability is not a general harm score.

```python
harm = ss.test_harm(
    source=source_scores,
    target=target_scores,
    worse="higher",
    rng=rng,
)
print(f"Harm p-value: {harm.pvalue:.4f}")  # → 0.0002
```

The small p-value is evidence that target scores appear more often in the high
tail. It signals a membership shift, not worse outcomes on its own.

### 5. Use a score tied to your outcome

For a real harm analysis, choose a score tied to your outcome—such as risk,
error, confidence, or quality. Use the table to match that score to its
harmful direction.

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

Use `test_shift` to test for rank-based separation in either direction. Use
`test_harm` to test movement toward the harmful tail declared before testing.
Rank-based separation is not, by itself, evidence of harm.

Next:

- [Is the new drug good enough?](../trials/check-drug-efficacy.md): the same test on 70 trial scores, with no model.
- [Monitor a credit model](../credit/monitor-credit.md): risk, confidence, and error on a HELOC book.
- [How the harm test works](../../explanation/harmful-shift-statistic.md): why the weighted AUC leans on the harmful tail.
