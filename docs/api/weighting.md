# Weighting parameters

Pass pre-computed or context-aware weights directly to `test_shift` or
`test_adverse_shift` to correct for known covariate shift between your source
and target groups.

## Choosing an approach

| Scenario | How to proceed |
|----------|-----------------|
| No weighting (default) | Omit `weights` |
| You have explicit per-sample weights | Pass `weights=my_weights` |
| You have classifier membership probabilities | Build weights with `contextual_weights(...)`, then pass `weights=` |

```python
import numpy as np
import samesame
from samesame.weights import contextual_weights

# No weighting (default)
result = samesame.test_shift(source=source_scores, target=target_scores)

# Explicit per-sample weights you computed yourself
result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    weights=my_weights,
)

# Context-aware weights derived from membership probabilities
weights = contextual_weights(
    source_prob=source_membership_probs,  # probabilities for source samples
    target_prob=target_membership_probs,  # probabilities for target samples
    mode="source",
)
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    weights=weights,
)
```

See [Sample weights](importance_weights.md) for the full `contextual_weights` reference
and guidance on choosing `mode` and `lambda_`.

For a step-by-step worked example, see the tutorial
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the conceptual background on why density ratios need stabilisation and when to choose
each mode, see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).
