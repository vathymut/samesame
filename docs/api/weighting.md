# Importance weights

Reframe the comparison around **common support** — the regions both groups share. If source and target barely overlap, reweight. Otherwise, don't.

## Scope

Reference for weighting utilities. For when and how to weight, see [Weight for common support](../how-to/weight-for-common-support.md); for the statistic they modify, see [How the harm test works](../explanation/harmful-shift-statistic.md). Concepts: [Core concepts](../explanation/core-concepts.md).

Start unweighted. An unweighted comparison keeps all regions both groups occupy. When overlap is poor, a few points can dominate. Weighting reframes around common support — it creates no information and changes the population. Not a default correction.

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

Weights preserve nominal size (`Σw = n` per group; `1` if unweighted) — they change influence, not classifier quality. Use them when poor overlap would let a few points dominate; otherwise omit `weights`.

## Domain weights

Pass separate `P(target|x)` arrays aligned with your scores (estimate out of sample). Choose which group(s) to reweight:

--8<-- "snippets/reweight-table.txt"

Shrinkage `λ` trades bias vs. variance — start at `0.5` and check `ESS/n` before going more aggressive:

--8<-- "snippets/shrinkage-table.txt"

## Effective sample size

```python
ess = weights.effective_sample_size()  # Kish (1965): (sum w)² / sum w²
print(ess.source, ess.target)  # compare each to its n
```

ESS translates unequal weights into equally-weighted counts (Kish 1965: `(Σw)²/Σw²`). Uniform → `ESS=n`; concentrated → `≈1`. Compare `ess.source` to `len(source)` and `ess.target` to `len(target)`.

`ESS/n` well below `0.5` warns the result rests on few points — no universal cutoff (the `n/4` heuristic has no published threshold; Elvira et al., 2022). If low even at `shrinkage=0.5`, groups lack common support; keep the comparison unweighted and report the unweighted p-value.

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
