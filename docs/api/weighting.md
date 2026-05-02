# Weighting strategies

Use this page when you need to pass a weighting strategy to `ShiftOptions` or
`AdverseShiftOptions` to correct for known covariate shift between your source
and target groups.

## Choosing a strategy

| Strategy | When to use it |
|----------|----------------|
| `NoWeighting` | Default. All samples are treated equally. |
| `SampleWeighting` | You already have computed per-sample weights. |
| `ContextualRIWWeighting` | You have membership probabilities from a classifier and want the library to compute RIW weights for you. |

```python
from samesame.weighting import ContextualRIWWeighting, NoWeighting, SampleWeighting
from samesame.advanced import AdverseShiftOptions, ShiftOptions
import samesame.advanced as advanced

# No weighting (default — identical to omitting the weighting argument)
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(weighting=NoWeighting()),
)

# Explicit per-sample weights you computed yourself
result = advanced.test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(weighting=SampleWeighting(values=my_weights)),
)

# Context-aware RIW weighting derived from membership probabilities
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
```

See [Importance weights](importance_weights.md) if you need to compute or inspect
raw importance weights outside of a test call.

::: samesame.weighting
