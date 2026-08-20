# Importance weights

Use `samesame.weights` when you want a shift test to focus on the part of feature space that source
and target actually share.

## Choose an approach

| Situation | What to do |
|-----------|------------|
| No weighting needed | Omit `weights` |
| You already have sample weights | Wrap them in `ImportanceWeights(source=..., target=...)` |
| You have domain-classifier probabilities | Build weights with `from_domain_probabilities(...)` |

```python
import samesame as ss
from samesame.weights import ImportanceWeights, from_domain_probabilities

result = ss.detect_shift(source_scores, target_scores)

result = ss.detect_shift(
    source_scores,
    target_scores,
    weights=ImportanceWeights(
        source=source_weights,
        target=target_weights,
    ),
)

weights = from_domain_probabilities(
    source_prob=source_domain_probs,
    target_prob=target_domain_probs,
    mode="both",
)

result = ss.detect_harm(
    source_scores,
    target_scores,
    direction=ss.Direction.HIGHER_IS_WORSE,
    weights=weights,
)
```

## When `from_domain_probabilities(...)` helps

Use it when source and target do not overlap well and you want the comparison to emphasize common
support rather than low-overlap outliers.

It takes three main controls:

- `source_prob` and `target_prob`, passed separately
- `mode`, which decides whether to reweight source, target, or both (default `"both"`)
- `lambda_`, which trades off correction strength against stability

## Choosing a mode

| Mode | What it emphasizes |
|------|--------------------|
| `mode="source"` | overlap from the source side |
| `mode="target"` | overlap from the target side |
| `mode="both"` | common support from both sides (default) |

`lambda_=0.5` is a practical default. Lower values correct more aggressively. Higher values move
closer to uniform weights.

## Effective sample size

Once you have `ImportanceWeights`, call `.effective_sample_size()` to get Kish's ESS for each group.
ESS is bounded above by the sample size; it drops toward 1 when a few observations carry almost all
the weight. Use it to decide whether your weights are trustworthy before you pass them to a test.

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
