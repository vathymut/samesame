# Advanced controls

Use this page when you need additional controls such as sample weights, more resamples,
the null distribution, or Bayesian evidence.

## What you get back

- `advanced.test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `advanced.test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, `.null_distribution`, and optional `.bayes_factor` and `.posterior`
- Both detailed result objects provide `.summary()` to get the simpler primary result

## Configuring tests with options objects

All advanced controls are passed through immutable options dataclasses instead of keyword arguments.
Weighting strategy types live in `samesame.weighting`; options types live in `samesame.advanced`.

```python
import samesame.advanced as advanced
from samesame.advanced import AdverseShiftOptions, ShiftOptions
from samesame.weighting import ContextualRIWWeighting, NoWeighting, SampleWeighting

# Custom number of resamples and a one-sided alternative
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(n_resamples=4999, alternative="greater"),
)

# Explicit per-sample weights
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(weighting=SampleWeighting(values=my_weights)),
)

# Context-aware RIW weighting for covariate shift adaptation
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

# Bayesian evidence alongside the permutation p-value
result = advanced.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    options=AdverseShiftOptions(bayesian=True),
)
print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {result.bayes_factor:.2f}")
```

## Reproducibility

Pass a `numpy.random.Generator` via `rng` to make any test deterministic:

```python
import numpy as np
from samesame.advanced import ShiftOptions

result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(rng=np.random.default_rng(42)),
)
```

::: samesame.advanced

