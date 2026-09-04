# Get started

In this tutorial, you will build a minimal monitoring workflow: create one score
per observation, test for a difference between source and target, and test
whether the change moved in a declared harmful direction. By the end, the
two-question habit — *did it change? did it get worse?* — should feel as
natural as checking a pulse.

In this tutorial, **source** is the reference distribution, such as training
data or a past deployment, and **target** is the distribution you want to
evaluate, typically the current deployment. Compare them using one
interpretable score for each observation, such as predicted risk, prediction
error, confidence, or an outlier score.

- **Any shift?** Use `ss.test_shift` — a broad, two-sided screen for whether the score distinguishes source from target; an AUC of `0.5` is chance.
- **Harmful shift?** Use `ss.test_harmful_shift(..., worse="higher"|"lower")` — a focused, one-sided test for whether the target moved toward the harmful tail you specify (`worse="lower"` flips the sign).

The two tests can reach different conclusions because a shift may be
harmless, harmful, or even beneficial depending on where it occurs. See
[How the harm test works](../../explanation/harmful-shift-statistic.md) for the
intuition and formula.

Interpret `.pvalue` alongside the statistic and the distributions: a small
p-value indicates evidence against label exchangeability, i.e. the assumption
that the two samples can be swapped, not the size or cause of the change.

## 1 — Make source and target

```python
import numpy as np
rng = np.random.default_rng(12345)
source = rng.normal(loc=0.0, scale=1.0, size=(400, 4))
target = rng.normal(loc=[0.7, 0.0, 0.0, 0.0], scale=1.0, size=(400, 4))
X = np.vstack([source, target])
labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
```

In production, the source and target should represent the two populations
relevant to your monitoring question - for example, training data versus a
current deployment. Choose the source carefully: every conclusion describes
the target relative to that reference.

## 2 — Score out of sample

Each row must be scored by a model that did not see that row. Here, a domain
classifier estimates `P(target|x)`, the probability that the row belongs to the
target rather than the source. Differences in this probability indicate how
well the features distinguish the two distributions, making it a useful
generic score for detecting *any* shift:

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

domain_prob = cross_val_predict(
    HistGradientBoostingClassifier(random_state=12345),
    X, labels, cv=10, method="predict_proba",
)[:, 1]
```

!!! warning "Honest scores"
    `samesame` only sees scores, not how they were made. If scores come from a fitted model, generate them out of sample using `cross_val_predict`, `oob_decision_function_`, or a held-out set. In-sample predictions let the model separate source from target using information it has already seen, which can produce misleading results and invalidate the test.

When testing a score with a clear interpretation of what constitutes a good or
bad outcome, keep it separate from the domain probability. Domain probability
describes distributional membership, not outcome quality. Use it to build
weights, and use the interpretable score as the harm score.

## 3 — Did anything change?

```python
import samesame as ss
source_scores = domain_prob[labels == 0]
target_scores = domain_prob[labels == 1]
shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
print(f"AUC {shift.statistic:.3f} p={shift.pvalue:.4f}")  # → 0.611, 0.0002
```

An AUC near `0.5` indicates little separation, while values farther from `0.5`
indicate stronger separation. Because `test_shift` is two-sided, separation in
either direction - for example, `0.8` or `0.2` - can reject the null.

## 4 — Did it get worse?

This is the question with a stake attached: not *did it change*, but *did it
move toward the outcomes you fear*. Before running the test, declare whether
higher or lower scores represent worse outcomes. Pass this choice as a string
or as `ss.Worse`; the two forms are interchangeable.

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

- A small `test_shift` p-value provides evidence that the score distributions
  differ.
- A small `test_harmful_shift` p-value provides evidence that the target shifted
  toward the harmful tail you declared, not merely that some shift occurred.

The test is directional by design: changing `worse` changes the harmful
direction being tested. Decide whether higher or lower values are worse before
looking at the result; do not choose the direction based on whichever gives a
smaller p-value.

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` to make the permutation-based p-values reproducible. The default is `n_resamples=9999`; `999` is useful while exploring, while `19999` gives better resolution for p-values below `0.001`.

For the harm test told on real data with no model at all, read [Is the new drug
good enough?](../trials/check-drug-efficacy.md). If source and target have poor
overlap in feature space, see [Weight for common
support](../weighting/weight-for-common-support.md) to learn how to reweight
the comparison.
