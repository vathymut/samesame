# Importance weights

## Why weighting exists

The unweighted comparison describes the full source and target samples. That
is appropriate when the tails are part of the population you want to compare.
It can be misleading when the groups barely overlap: a handful of source-only
or target-only cases can dominate the statistic even though there is little
evidence about the other group's behaviour there.

Importance weighting changes the question to one about **common support** -
the regions of feature space represented by both groups. This can make the
comparison more stable, but it does not create information outside the
overlap and it is not a default correction. Start unweighted and use weights
when you have a substantive overlap concern.

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

Weights are normalised to sum to group size, so the two groups retain their
original nominal sample sizes in the permutation calculation. An inactive
group gets weight `1`. The weights affect which observations count more; they
do not repair a poorly estimated domain classifier.

## `domain_weights`

- `source` / `target` - domain probabilities `P(target|x)` as **separate** 1-D arrays in `[0, 1]`, aligned to scores. These probabilities should be out of sample or otherwise honestly estimated. The prior ratio `n_source/n_target` is inferred from lengths. Values are clipped to `[1e-6, 1 - 1e-6]` before ratios to avoid infinities.
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

Effective sample size (ESS) translates unequal weights into an approximate
number of equally weighted observations. Uniform weights give `ESS = n`; if
one or two observations carry most of the mass, ESS approaches `1`.

Treat `ESS < n/4` as a warning rather than a cutoff. A low ESS means the
weighted result is fragile and largely driven by a few observations. If ESS
remains low after moderate shrinkage, the groups may not have enough common
support for a reliable weighted comparison.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
