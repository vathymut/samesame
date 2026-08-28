# Importance weights

When a plain comparison gives too much influence to points the other group rarely contains, reframe to **common support** via weighting.

> Source: `src/samesame/weights.py` · `src/samesame/_permutation.py`

## When to use

| Situation | What to do |
|-----------|------------|
| No overlap concern | Omit `weights` |
| You have sample weights | `ss.ImportanceWeights(source=..., target=...)` |
| You have `P(target\|x)` | `ss.domain_weights(source=..., target=...)` |

```python
import samesame as ss

# no weighting
ss.test_shift(source=source_scores, target=target_scores, rng=12345)

# custom weights
ss.test_shift(
    source=source_scores, target=target_scores,
    weights=ss.ImportanceWeights(source=source_weights, target=target_weights),
    rng=12345,
)

# from a domain classifier
weights = ss.domain_weights(
    source=source_domain_prob,  # P(target | x) for source
    target=target_domain_prob,  # P(target | x) for target
    reweight="both", shrinkage=0.5,
)
ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=weights, rng=12345)
```

## `domain_weights`

- `source` / `target` — domain probabilities `P(target|x)` as **separate** 1-D arrays in `[0, 1]`, aligned to scores. Prior `n_source/n_target` is inferred from lengths.
- `reweight` — `"source"`, `"target"`, or `"both"` (default `"both"`). String or `ss.ReweightMode`.
- `shrinkage` λ in `[0, 1]` — `0` = plain ratio, `1` = uniform. Default `0.5`.

--8<-- "snippets/reweight-table.txt"

Start at `0.5`; lower is stronger but higher variance. Inactive groups get weight `1` (uniform).

--8<-- "snippets/clipping-note.txt"

See [Weight for common support](../examples/weighting/weight-for-common-support.md) for when to weight and how to diagnose.

## Effective sample size

```python
ess = weights.effective_sample_size()
print(ess.source, ess.target)  # compare each to its n
```

ESS ≤ `n`; →1 when a few points dominate. --8<-- "snippets/ess-rule.txt"

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
