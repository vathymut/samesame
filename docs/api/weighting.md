# Weights

`samesame.weights` is the low-level weighting Module for the Source and Target score-comparison seam.
Use it when you already have Source and Target score arrays and want weighted testing through
`shift.detect_shift(...)` or `shift.detect_harm(...)`.

## Choosing an approach

| Scenario | How to proceed |
|----------|-----------------|
| No weighting (default) | Omit `weights` |
| You already have sample weights | Wrap them in `ImportanceWeights(source=..., target=...)`, then pass `weights=` |
| You have Domain Probabilities from a domain classifier | Build weights with `from_domain_probabilities(...)`, then pass `weights=` |

```python
import samesame as ss
from samesame.weights import ImportanceWeights, from_domain_probabilities

# No weighting (default)
result = ss.shift.detect_shift(source_scores, target_scores)

# Sample weights computed in your own adapter
result = ss.shift.detect_shift(
    source_scores,
    target_scores,
    weights=ImportanceWeights(
        source=source_weights,
        target=target_weights,
    ),
)

# Importance weights derived from domain probabilities
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
```

## When to use `from_domain_probabilities`

Use importance weights when Source and Target have different feature distributions and you want the
test to focus on the region where both groups overlap rather than on low-overlap outliers. If you do
not already know you have covariate shift, leave `weights` unset.

The `from_domain_probabilities(...)` adapter keeps the Interface small:

- `source_prob` and `target_prob` are passed separately, so the Source-first ordering invariant is structural.
- `mode` chooses whether to reweight Source, Target, or both groups.
- `lambda_` stabilises density-ratio weighting by blending toward uniform weights.

## Choosing a mode

| Mode | What it does |
|------|--------------|
| `mode="source"` | Down-weights Source samples foreign to Target. Target samples keep unit weight. |
| `mode="target"` | Down-weights Target samples foreign to Source. Source samples keep unit weight. |
| `mode="both"` | Down-weights outliers in both groups and focuses the test on common support. |

`lambda_` controls numerical stability: `0.0` is the plain density ratio and `1.0` is uniform weights.
The default `0.5` is a practical starting point.

Weights for each active group are automatically normalized to sum to that group's sample size. In
`mode="both"`, Source and Target are normalized independently. Non-active groups always receive unit
weights.

For a step-by-step worked example, see
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the conceptual background on why density ratios need stabilisation and when to choose each mode,
see [Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).

## API reference

::: samesame.weights
