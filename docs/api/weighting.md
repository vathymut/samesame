# Importance weights

## Why weighting exists

An unweighted comparison describes the full source and target samples, including regions that only one group occupies. That is appropriate when those regions belong to the populations you want to compare. When the groups barely overlap, however, a small number of observations from those regions can dominate the statistic even though the data contain little information about how the other group would behave there.

Importance weighting reframes the comparison around **common support** — the regions represented by both groups. It can make the test more stable, but it does not create information where the groups do not overlap, and it changes the population the test describes. It is not a default correction. Start without weights and introduce them only when poor feature overlap is a real concern.

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

Weights are normalized to sum to each group's size, so the groups keep their original nominal sample sizes in the permutation. A group that is not reweighted receives a weight of `1` for every observation. Weights change how much each observation counts; they do not compensate for a poorly estimated domain classifier.

## Domain weights

Pass separate one-dimensional arrays of domain probabilities `P(target|x)`, each aligned with the scores you intend to test. Estimate these probabilities honestly — out of sample when a model produces them. Choose which group or groups to reweight:

--8<-- "snippets/reweight-table.txt"

Shrinkage `λ` controls the bias–variance trade-off of the correction. Start at the default and check `ESS/n` before making the correction more aggressive:

--8<-- "snippets/shrinkage-table.txt"

See [Weight for common support](../examples/weighting/weight-for-common-support.md) for guidance on diagnosing overlap and weight concentration.

## Effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(ess.source, ess.target)  # compare each to its n
```

Effective sample size (ESS) translates unequal weights into an approximate count of equally weighted observations (Kish, 1965: `(sum w)² / sum w²`). Uniform weights yield `ESS = n`; when one or two observations carry most of the weight, ESS falls toward `1`.

Compare each ESS to its `n` through `ESS/n`. A low ratio — for example well below `0.5` — signals that the weighted result is fragile and driven by a few observations. This is a warning, not a hard validation rule, and there is no universal cutoff from Kish. The often-quoted `ESS < n/4` is a rough illustrative heuristic with no published empirical threshold (see Elvira et al., 2022). If `ESS/n` stays low even at `shrinkage=0.5`, the groups may lack enough common support for a reliable weighted comparison; consider keeping the comparison unweighted.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
