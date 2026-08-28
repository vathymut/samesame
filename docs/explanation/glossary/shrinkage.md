# Shrinkage

`shrinkage` (λ) in `[0, 1]` — RIW (Relative Importance Weight) parameter trading correction strength against stability (Yamada et al. 2013).

| `shrinkage` | Effect |
|-------------|--------|
| `0.0` | Plain density ratio. Strongest correction, highest variance. |
| `0.5` | Default. Good balance. |
| `1.0` | Uniform weights. No correction. |

Lower = more aggressive; higher = more conservative. Start at `0.5` and check ESS before lowering.

RIW formulas (Yamada) — with $\hat{p} = P(\text{target} \mid x)$ estimated by the domain classifier and $\hat{r}(x) = \hat{p}(x)/(1-\hat{p}(x)) \cdot n_{\text{source}}/n_{\text{target}}$:

- Source: $w_{\text{source}}(x) = \hat{r}(x) / ((1-\lambda) + \lambda \hat{r}(x))$
- Target: $w_{\text{target}}(x) = 1 / (\lambda + (1-\lambda) \hat{r}(x))$

You don't compute these — `ss.domain_weights(..., shrinkage=λ)` does.

If ESS collapses at `λ=0` (see [Diagnose weight concentration](../../examples/weighting/diagnose-weight-concentration.md)), raise `λ` rather than trusting a concentrated weight.

See also: [ESS](ess.md), [Reweight](reweight.md), [When importance weights help](../importance-weights-rationale.md).
