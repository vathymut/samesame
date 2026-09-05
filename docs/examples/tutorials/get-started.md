# Get started

In this tutorial you will build a minimal monitoring workflow: create one score per observation, check whether source and target differ, and then check whether the change moved toward a harmful direction you declare. By the end, the two-question habit — *did it change? did it get worse?* — should feel familiar.

--8<-- "snippets/source-target.txt"

Compare source and target with one interpretable score per observation, such as predicted risk, prediction error, confidence, or an outlier score.

- **Any shift?** Use `ss.test_shift` — a broad, two-sided screen for whether the score separates source from target. An AUC of `0.5` means no separation.
- **Harmful shift?** Use `ss.test_harmful_shift(..., worse="higher"|"lower")` — a focused, one-sided test for whether the target moved toward the harmful tail you specify. Use `worse="lower"` when smaller values mean more harm.

The two tests can reach different conclusions because a distributional shift may be benign, harmful, or even beneficial depending on where it falls. See [How the harm test works](../../explanation/harmful-shift-statistic.md) for intuition and the formula.

Interpret `.pvalue` alongside the statistic and the score distributions.

--8<-- "snippets/pvalue-caveat.txt"

## 1 — Make source and target

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, source and target should reflect the two populations that define your monitoring question — for example, training data versus the current deployment. Choose the source with care: every conclusion describes the target relative to that reference.

## 2 — Score out of sample

Each row should be scored by a model that did not see that row during training. Here, a domain classifier estimates `P(target|x)`, the probability that a row belongs to the target rather than the source. When this probability separates the two groups well, it signals a distributional difference, which makes it a useful generic score for detecting *any* shift:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

!!! warning "Honest scores"
    `samesame` only sees scores, not how they were made.

    --8<-- "snippets/honest-scores.txt"

When you test a score with a clear notion of better and worse, keep it separate from the domain probability. Domain probability describes group membership, not outcome quality. Use it to build weights when needed, and use the interpretable score as the harm score.

## 3 — Did anything change?

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"AUC {shift.statistic:.3f} p={shift.pvalue:.4f}")  # → 0.611, 0.0002
```

An AUC near `0.5` means the score hardly separates the groups, while values well above or below `0.5` mean stronger separation. Because `test_shift` is two-sided, separation in either direction — for example `0.8` or `0.2` — can lead to rejection.

## 4 — Did it get worse?

The more important question is whether the change moved toward outcomes you want to avoid.

--8<-- "snippets/worse-declaration.txt"

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

- A small `test_shift` p-value is evidence that the score distributions differ.
- A small `test_harmful_shift` p-value is evidence that the target shifted toward the harmful tail you declared.

The test is directional by design: changing `worse` changes the tail under examination.

--8<-- "snippets/worse-tip.txt"

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` to make permutation p-values reproducible. The default is `n_resamples=9999`; use `999` while exploring and `19999` when you need finer resolution for p-values below `0.001`.

## Recap

You created a score for each observation, checked for any distributional shift with `test_shift`, and then tested a specific harmful direction with `test_harmful_shift`. The first tells you whether the score separates source from target; the second tells you whether the target moved toward the tail you care about.

For the same harm test on real trial data with no model at all, see [Is the new drug good enough?](../trials/check-drug-efficacy.md). If source and target overlap poorly in feature space, see [Weight for common support](../weighting/weight-for-common-support.md) to learn how to reweight the comparison.
