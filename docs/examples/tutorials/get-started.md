# Get started

This tutorial shows how to turn scores into two answers: *did it change?* and *did it get worse?*

## Prerequisites

- Python 3.12+ with `numpy`, `scipy`, `scikit-learn`, and `samesame` installed.
- Comfort with p-values and training a classifier (you will use `cross_val_predict` once).

--8<-- "snippets/source-target.txt"

## Steps

The two tests answer different questions. `ss.test_shift` is a broad, two-sided test for any difference between source and target. `ss.test_harmful_shift(..., worse="higher"|"lower")` is a focused, one-sided test for detecting a harmful shift. You can apply both tests to the score that represents the outcome you care about.

### 1. Create source and target

Start with two datasets. Here, source represents a sample from the reference population and target represents a sample from the current population. In practice, source might be training data or a past deployment.

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

With this p-value, we reject the null of no shift, as expected.

### 4. Did it get worse?

The domain probability can also serve as an outlier score when target-like observations represent the harmful direction: a higher domain probability means the observation looks less like the reference sample, so use `worse="higher"` when these deviations are harmful.

```python
harm = ss.test_harmful_shift(
    source=source_scores,
    target=target_scores,
    worse="higher",
    rng=rng,
)
print(f"Harm p-value: {harm.pvalue:.4f}")  # → 0.0002
```

Here again, we reject the null of no harmful shift, meaning the target sample often does not resemble the source sample.

### 5. Examples of harmful scores

The domain probability is one way to define an outlier score. You can also test harmful shift using a score tied directly to the outcome you care about. The examples below show two possibilities: risk or error, where higher scores are harmful, and confidence or quality, where lower scores are harmful. Replace them with your own score if need be. See [Shift testing](../../api/testing.md) for the test signatures and polarity options.

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

## Recap

You now have one score and two tests. `test_shift` tells you whether source and target differ at all. `test_harmful_shift` tells you whether target moved toward the harmful tail you specified before testing. Keeping those questions separate prevents a detectable shift from being mistaken for harmful shift.

Where you'd like to go next:

- [Is the new drug good enough?](../trials/check-drug-efficacy.md): the same test on 70 trial scores, with no model to fit.
   - [Monitor a credit model](../credit/monitor-credit.md): apply the tests to a HELOC risk-monitoring example.
- [How the harm test works](../../explanation/harmful-shift-statistic.md): why the weighted AUC leans into the harmful tail.
