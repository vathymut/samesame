# Importance weights

When points the other group rarely contains would dominate, reframe to **common support** via weighting.

> Source: `src/samesame/weights.py` · `src/samesame/_permutation.py`

## When to use

| Situation | What to do |
|-----------|------------|
| No overlap concern | Omit `weights` |
| You have sample weights | `ss.ImportanceWeights(source=..., target=...)` |
| You have `P(target\|x)` | `ss.domain_weights(source=..., target=...)` |

```python
import samesame as ss

ss.test_shift(source=source_scores, target=target_scores, rng=12345)
ss.test_shift(source=source_scores, target=target_scores,
              weights=ss.ImportanceWeights(source=source_weights, target=target_weights), rng=12345)

weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=weights, rng=12345)
```

Weights are normalised to sum to group size; inactive groups get weight `1`.

## `domain_weights`

- `source` / `target` — domain probabilities `P(target|x)` as **separate** 1-D arrays in `[0, 1]`, aligned to scores. Prior `n_source/n_target` is inferred from lengths. Clipped to `[1e-6, 1 − 1e-6]` before ratios to avoid infinities.
- `reweight` — which group(s) to reweight (default `"both"`):

--8<-- "snippets/reweight-table.txt"

- `shrinkage` λ in `[0, 1]` — `0` = plain density ratio (strong, high variance), `1` = uniform (no correction). Default `0.5`. Start there and check ESS before lowering:

--8<-- "snippets/shrinkage-table.txt"

See [Weight for common support](../examples/weighting/weight-for-common-support.md) for diagnosis.

## Effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(ess.source, ess.target)  # compare each to its n
```

Uniform → `ESS = n`; concentrated → `≈ 1`. Worry when `ESS < n/4` for either group (rule of thumb, not cutoff) — with healthy ESS a significant weighted result is convincing; with `ESS ≈ 1` it's driven by a few points.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
