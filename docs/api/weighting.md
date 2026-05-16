# Weighting parameters

Pass pre-computed or domain-probability-derived weights directly to
`shift.detect_shift` or `shift.detect_harm` to correct for known covariate shift between your source
and target groups.

## Choosing an approach

| Scenario | How to proceed |
|----------|-----------------|
| No weighting (default) | Omit `weights` |
| You have sample weights | Wrap in `ImportanceWeights(source=..., target=...)`, then pass `weights=` |
| You have domain probabilities from a domain classifier | Build weights with `from_domain_probabilities(...)`, then pass `weights=` |

```python
import numpy as np
import samesame as ss
from samesame.weights import from_domain_probabilities

# No weighting (default)
result = ss.shift.detect_shift(source_scores, target_scores)

# Sample weights computed yourself
result = ss.shift.detect_shift(
    source_scores,
    target_scores,
    weights=ss.weights.ImportanceWeights(
        source=source_weights,
        target=target_weights,
    ),
)

# Importance weights derived from domain probabilities
weights = from_domain_probabilities(
    source_prob=source_domain_probs,  # domain probabilities for source samples
    target_prob=target_domain_probs,  # domain probabilities for target samples
    mode="source",
)
result = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
    weights=weights,
)
```

See [Sample weights](importance_weights.md) for the full `from_domain_probabilities` reference
and guidance on choosing `mode` and `lambda_`.

For a step-by-step worked example, see the tutorial
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the conceptual background on why density ratios need stabilisation and when to choose
each mode, see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).
