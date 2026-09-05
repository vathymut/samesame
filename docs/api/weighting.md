# Importance weights

## Why weighting exists

An unweighted comparison describes the full source and target samples, including
regions seen by only one group. This is appropriate when those regions are part
of the populations you want to compare. But when the groups barely overlap, a
few observations from these regions can dominate the statistic even though the
data provide little evidence about the other group's behavior there.

Importance weighting changes the question to one about **common support** — the
regions represented by both groups. It can make the comparison more stable, but
it does not create information where groups do not overlap, and it changes the
population the test describes. It is not a default correction. Start unweighted
and use weights only when poor feature overlap is a real concern.

??? details "Source files"
    `src/samesame/weights.py` · `src/samesame/_permutation.py`

## When to use

| Situation | Recommendation |
|-----------|----------------|
| No overlap concern | Omit `weights` |
| You have sample weights | Pass `ss.ImportanceWeights(source=..., target=...)` |
| You have `P(target\|x)` | Build weights with `ss.domain_weights(source=..., target=...)` |

```python
import numpy as np
import samesame as ss

ss.test_shift(source=source_scores, target=target_scores,
              rng=np.random.default_rng(12345))
ss.test_shift(source=source_scores, target=target_scores,
              weights=ss.ImportanceWeights(source=source_weights, target=target_weights),
              rng=np.random.default_rng(12345))

weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", weights=weights,
                      rng=np.random.default_rng(12345))
```

Weights are normalized to sum to each group's size, so the groups retain their
original nominal sample sizes in the permutation calculation. An inactive group
gets weight `1`. Weights change which observations count more; they do not
repair a poorly estimated domain classifier.

## Domain weights

Pass separate 1-D arrays of domain probabilities `P(target|x)`, aligned to the
scores you intend to test; estimate them honestly — out of sample if a model
produces them. Which group(s) to reweight:

--8<-- "snippets/reweight-table.txt"

Shrinkage λ controls the bias–variance tradeoff of the correction — start at
the default and check `ESS/n` before going more aggressive:

--8<-- "snippets/shrinkage-table.txt"

See [Weight for common support](../examples/weighting/weight-for-common-support.md) for guidance on diagnosing overlap and weight concentration.

## Effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(ess.source, ess.target)  # compare each to its n
```

Effective sample size (ESS) translates unequal weights into an approximate
number of equally weighted observations (Kish, 1965: ``(sum w)² / sum w²``).
Uniform weights give `ESS = n`; when one or two observations carry most of the
mass, ESS approaches `1`.

Compare each ESS to its ``n`` via ``ESS/n``. A low ratio (e.g., substantially
below ``0.5``) warns that the weighted result is fragile and largely driven by
a few observations; it is a warning, not a hard validation rule, and there is
no universal cutoff from Kish. The often-quoted ``ESS < n/4`` is only a rough
illustrative heuristic with no published empirical threshold (see Elvira et
al., 2022). If ``ESS/n`` remains low even at ``shrinkage=0.5``, the groups may
not have enough common support for a reliable weighted comparison; consider
leaving the comparison unweighted.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
