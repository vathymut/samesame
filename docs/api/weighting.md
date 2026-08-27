# Importance weights

Use `samesame.weights` when a plain comparison gives too much influence to observations that the
other group rarely contains. Weighting changes the question: you are comparing the groups mainly
where they overlap.

## Choose an approach

| Situation | What to do |
|-----------|------------|
| No weighting needed | Omit `weights` |
| You already have sample weights | Wrap them in `ImportanceWeights(source=..., target=...)` |
| You have domain-classifier probabilities | Build weights with `domain_weights(...)` |

```python
import samesame as ss
from samesame.weights import ImportanceWeights, domain_weights

result = ss.test_shift(source_scores, target_scores)

result = ss.test_shift(
    source_scores,
    target_scores,
    weights=ImportanceWeights(
        source=source_weights,
        target=target_weights,
    ),
)

weights = domain_weights(
    source=source_domain_probs,
    target=target_domain_probs,
    reweight="both",
)

result = ss.test_harmful_shift(
    source_scores,
    target_scores,
    worse="higher",
    weights=weights,
)
```

## When `domain_weights(...)` helps

Use it when source and target do not overlap well and you want the comparison to emphasize common
support rather than low-overlap observations.

It takes two main controls:

- `source` and `target`, passed separately as domain-classifier probabilities that estimate
  `P(target | observation)`. These are not the raw source and target observations or test scores.
- `reweight`, which decides whether to reweight source, target, or both (default `"both"`)
- `shrinkage`, which trades off correction strength against stability

Probabilities must be in `[0, 1]`. Values at 0 or 1 are automatically clipped away
from the boundaries before weights are calculated.

## Choosing what to reweight

| Mode | What it emphasizes |
|------|--------------------|
| `reweight="source"` | overlap from the source side |
| `reweight="target"` | overlap from the target side |
| `reweight="both"` | common support from both sides (default) |

`shrinkage=0.5` is a practical default. Lower values correct more aggressively. Higher values produce
weights closer to uniform.

## Effective sample size

Once you have `ImportanceWeights`, call `.effective_sample_size()` to get Kish's ESS for each group.
ESS is bounded above by the sample size; it drops toward 1 when a few observations carry almost all
the weight. Use it to assess whether the weights are sufficiently dispersed for the comparison.

```python
ess = weights.effective_sample_size()
print(ess.source, ess.target)
```

For a worked example, see
[Diagnose weight concentration with effective sample size](../examples/weighting/diagnose-weight-concentration.md).

For a worked example on building weights, see
[Focus on shared support with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the intuition behind the formulas, see
[When importance weights help](../explanation/importance-weights-rationale.md).

## API

::: samesame.weights
