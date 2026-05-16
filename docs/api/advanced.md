# Additional controls

Both `shift.detect_shift` and `shift.detect_harm` accept keyword arguments for
resampling and weighting. All results include the full null distribution.
Bayesian evidence is available separately via `shift.infer_harm`.

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `shift.detect_harm(...)` returns `HarmResult` with `.statistic`, `.pvalue`, `.direction`, and `.null_distribution`
- `shift.infer_harm(...)` returns `HarmInference` with `.posterior` and `.bayes_factor`

## Configuring tests

All controls are direct keyword arguments — no wrapper objects required.

```python
import numpy as np
import samesame as ss
from samesame.weights import from_domain_probabilities

# Custom number of resamples and a one-sided alternative
result = ss.shift.detect_shift(
    source_scores,
    target_scores,
    n_resamples=4999,
    alternative="greater",
)

# Sample weights wrapped in ImportanceWeights
result = ss.shift.detect_shift(
    source_scores,
    target_scores,
    weights=ss.weights.ImportanceWeights(
        source=source_weights,
        target=target_weights,
    ),
)

# Importance weights from domain probabilities
weights = from_domain_probabilities(
    source_prob=source_domain_probs,
    target_prob=target_domain_probs,
    mode="source",
)
result = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
    weights=weights,
)

# Bayesian evidence alongside the permutation p-value
result = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
)
evidence = ss.shift.infer_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
)
print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {evidence.bayes_factor:.2f}")
```

## Reproducibility

Pass a seed or NumPy RNG via `random_state` to make any test deterministic:

```python
import numpy as np

result = ss.shift.detect_shift(
    source_scores,
    target_scores,
    random_state=42,
)
```

::: samesame.shift

## Bayes factor utilities

Use these functions to convert between p-values and Bayes factors, or to
compute Bayes factors directly from posterior draws returned by
`shift.infer_harm`.

::: samesame.shift.as_bf

::: samesame.shift.as_pvalue

::: samesame.shift.bayes_factor
