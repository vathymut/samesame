# Shrinkage

`shrinkage` (λ) in `[0, 1]` — RIW (Relative Importance Weight) parameter trading correction strength against stability (Yamada et al. 2013).

| `shrinkage` | Effect |
|-------------|--------|
| `0.0` | Plain density ratio. Strongest correction, highest variance. |
| `0.5` | Default. Good balance. |
| `1.0` | Uniform weights. No correction. |

Lower = more aggressive; higher = more conservative. Start at `0.5` and check ESS before lowering.

RIW formulas (Yamada):

- Source: `w_source = r / ((1-λ) + λ·r)`
- Target: `w_target = 1 / (λ + (1-λ)·r)`

where `r = p/(1-p) · n_source/n_target` and `p = P(target|x)`. You don't compute these — `ss.domain_weights(..., shrinkage=λ)` does.

If ESS collapses at `λ=0` (see [Diagnose weight concentration](../../examples/weighting/diagnose-weight-concentration.md)), raise `λ` rather than trusting a concentrated weight.

See also: [ESS](ess.md), [Reweight](reweight.md), [When importance weights help](../importance-weights-rationale.md).
