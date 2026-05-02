# Sample weights

Pass pre-computed or context-aware weights directly to `test_shift` or
`test_adverse_shift` to correct for known covariate shift between your source
and target groups.

## Choosing an approach

| Scenario | Parameters to use |
|----------|-------------------|
| No weighting (default) | Omit `weights` and `membership_prob` |
| You have explicit per-sample weights | `weights=my_weights` |
| You have classifier membership probabilities | `membership_prob=probs`, `mode=...` |

```python
import samesame

# No weighting (default)
result = samesame.test_shift(source=source_scores, target=target_scores)

# Explicit per-sample weights you computed yourself
result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    weights=my_weights,
)

# Context-aware weights derived from membership probabilities
result = samesame.test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    membership_prob=membership_probs,
    mode="source",
)
```

See [Sample weights](importance_weights.md) if you need to compute or inspect
weights outside of a test call using `samesame.weights.contextual_weights`.
