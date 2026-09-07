# Importance weights

Reframe your comparison around **common support**, the ground both groups share. When source and target barely overlap, reweight. Otherwise, leave weights out.

## Scope

Reference for weighting utilities. For when and how to weight, see [Weight for common support](../how-to/weight-for-common-support.md); for the statistic they modify, see [How the harm test works](../explanation/harmful-shift-statistic.md). Concepts: [Core concepts](../explanation/core-concepts.md).

Start unweighted. An unweighted comparison keeps every region both groups occupy. When overlap is poor, a few points can dominate, so weighting reframes the question around common support. It adds no information and changes the population you describe, which is why it isn't a default correction.

??? details "Source files"
    `src/samesame/weights.py` · `src/samesame/_permutation.py`

## Which weights?

| Situation | Action |
|-----------|----------------|
| No overlap concern | Omit `weights` |
| You have sample weights | Pass `ss.ImportanceWeights(source=..., target=...)` |
| You have `P(target|x)` | Build weights with `ss.domain_weights(source=..., target=...)` |

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

Weights preserve nominal size (`Σw = n` per group; `1` when unweighted). They change influence, not classifier quality. Reach for them when poor overlap would let a few points dominate; otherwise leave `weights` out.

## Domain weights

Pass separate `P(target|x)` arrays aligned with your scores (estimate out of sample). Choose which group(s) to reweight:

--8<-- "snippets/reweight-table.txt"

Shrinkage `λ` trades bias versus variance. Start at `0.5` and check `ESS/n` before going more aggressive:

--8<-- "snippets/shrinkage-table.txt"

## Effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(ess.source, ess.target)  # compare each to its n
```

ESS translates uneven weights into an equally weighted count (Kish 1965: `(Σw)²/Σw²`). Uniform weights give `ESS=n`; concentrated weights pull it toward `≈1`. Compare `ess.source` to `len(source)` and `ess.target` to `len(target)`.

`ESS/n` well below `0.5` warns your result rests on a few points. There is no universal cutoff (the `n/4` heuristic has no published threshold; Elvira et al., 2022). If ESS stays low even at `shrinkage=0.5`, the groups share too little ground. Keep the comparison unweighted and report the unweighted p-value.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
