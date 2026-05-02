# Additional controls

Both `test_shift` and `test_adverse_shift` accept keyword arguments for
resampling, weighting, and Bayesian evidence. All results include the full
null distribution.

## What you get back

- `test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, `.null_distribution`, and optional `.bayes_factor` and `.posterior`

## Configuring tests

All controls are direct keyword arguments — no wrapper objects required.

```python
import samesame

# Custom number of resamples and a one-sided alternative
result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    n_resamples=4999,
    alternative="greater",
)

# Explicit per-sample weights
result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    weights=my_weights,
)

# Context-aware weights from membership probabilities
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    membership_prob=membership_probs,
    mode="source",
)

# Bayesian evidence alongside the permutation p-value
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    bayesian=True,
)
print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {result.bayes_factor:.2f}")
```

## Reproducibility

Pass a `numpy.random.Generator` via `rng` to make any test deterministic:

```python
import numpy as np

result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    rng=np.random.default_rng(42),
)
```

::: samesame._api

