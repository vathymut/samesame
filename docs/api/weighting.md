# Importance weights

Use `samesame.weights` when a plain comparison gives too much influence to points the other group rarely contains. Weighting reframes to **common support**.

> Source: `src/samesame/weights.py` · `src/samesame/_permutation.py`

## When to use

| Situation | What to do |
|-----------|------------|
| No overlap concern | Omit `weights` |
| You already have sample weights | `ss.ImportanceWeights(source=..., target=...)` |
| You have `P(target\|x)` | `ss.domain_weights(source=..., target=...)` |

```python
import samesame as ss

# no weighting
result = ss.test_shift(source=source_scores, target=target_scores, rng=12345)

# custom weights
result = ss.test_shift(
    source=source_scores, target=target_scores,
    weights=ss.ImportanceWeights(source=source_weights, target=target_weights),
    rng=12345,
)

# from a domain classifier
weights = ss.domain_weights(
    source=source_domain_prob,  # P(target | x) for source
    target=target_domain_prob,  # P(target | x) for target
    reweight="both",
    shrinkage=0.5,
)
result = ss.test_harmful_shift(
    source=source_scores, target=target_scores, worse="higher", weights=weights, rng=12345,
)
```

--8<-- "snippets/clipping-note.txt"

## How `ss.domain_weights` works

- `source` and `target` — domain probabilities `P(target|x)` as **separate** 1-D arrays in `[0, 1]`, aligned to scores. Prior `n_source/n_target` inferred from lengths.
- `reweight` — `"source"`, `"target"`, or `"both"` (default `"both"`). String or `ss.ReweightMode`.
- `shrinkage` λ in `[0, 1]` — `0` = plain ratio, `1` = uniform. Default `0.5`.

## Choosing what to reweight

--8<-- "snippets/reweight-table.txt"

Inactive groups get weight `1` (uniform). Start at `0.5`; lower is stronger but higher variance. Check ESS — see [Weight for common support](../examples/weighting/weight-for-common-support.md).

## Effective sample size

```python
ess = weights.effective_sample_size()
print(ess.source, ess.target)  # compare each to its n
```

ESS ≤ `n`; `→1` when a few points dominate.

--8<-- "snippets/ess-rule.txt"

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
