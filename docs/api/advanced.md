# Additional controls

Both `test_shift` and `test_adverse_shift` accept keyword arguments for
resampling and weighting. All results include the full null distribution.
Bayesian evidence is available separately via `adverse_shift_posterior`.

## What you get back

- `test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, and `.null_distribution`
- `adverse_shift_posterior(...)` returns `BayesianEvidence` with `.posterior` and `.bayes_factor`

## Configuring tests

All controls are direct keyword arguments — no wrapper objects required.

```python
import numpy as np
import samesame
from samesame.weights import contextual_weights

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
weights = contextual_weights(
    source_prob=source_membership_probs,
    target_prob=target_membership_probs,
    mode="source",
)
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    weights=weights,
)

# Bayesian evidence alongside the permutation p-value
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
)
evidence = samesame.adverse_shift_posterior(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    result=result,
)
print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {evidence.bayes_factor:.2f}")
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

