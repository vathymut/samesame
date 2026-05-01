# Advanced controls

Use this page when you need additional controls such as sample weights, more resamples,
the null distribution, or Bayesian evidence.

## What you get back

- `advanced.test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `advanced.test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, `.null_distribution`, and optional `.bayes_factor` and `.posterior`
- Both detailed result objects provide `.summary()` if you want the simpler primary result

## Configuring tests with options objects

All advanced controls are passed through immutable options dataclasses instead of keyword arguments:

```python
from samesame import advanced
from samesame.advanced import (
    AdverseShiftOptions,
    ContextualRIWWeighting,
    NoWeighting,
    SampleWeighting,
    ShiftOptions,
)

# Custom resamples + alternative hypothesis
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(n_resamples=4999, alternative="greater"),
)

# Explicit sample weights
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(weighting=SampleWeighting(values=my_weights)),
)

# Context-aware RIW weighting
result = advanced.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    options=AdverseShiftOptions(
        weighting=ContextualRIWWeighting(
            probabilities=membership_probs,
            mode="source-reweighting",
        ),
    ),
)

# Bayesian evidence
result = advanced.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    options=AdverseShiftOptions(bayesian=True),
)
```

::: samesame.advanced
