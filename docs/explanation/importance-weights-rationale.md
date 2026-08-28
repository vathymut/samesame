# When importance weights help

Weighting helps when two things coincide:

- a real change in the region you care about
- plus points where the other group almost never goes (low overlap)

Without weighting, those low-overlap points can dominate a permutation test. Weighting reframes the comparison to **common support**.

> **Example:** training has many 20-year-old students production never sees; production has retirees never seen in training. Unweighted is swayed by extremes; weighted focuses on the 30–60 overlap.

For code, see [Weight for common support](../examples/weighting/weight-for-common-support.md) (synthetic + HELOC) and [API](../api/weighting.md).

## The density ratio and its instability

A domain classifier gives `p̂(x)=P(target|x)`. The standard correction is:

$$
\hat{r}(x) = \frac{\hat{p}(x)}{1-\hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

Prior ratio `n_source/n_target` is inferred from sizes, not tuned. Plain `r̂` is powerful but unstable — when groups separate well, a few points get huge weights. See [Weight for common support](../examples/weighting/weight-for-common-support.md).

--8<-- "snippets/clipping-note.txt"

## How `samesame` stabilises: RIW

Relative importance weighting (Yamada et al. 2013) blends the plain ratio toward uniform. You don't compute it — `ss.domain_weights` does. Lower `shrinkage` = stronger correction; higher = more stable; `0.5` is the default.

??? example "RIW formulas (experts)"
    With `r̂` above and `λ = shrinkage ∈ [0,1]`:
    `w_source = r̂ / ((1-λ)+λr̂)`, `w_target = 1 / (λ+(1-λ)r̂)`

--8<-- "snippets/shrinkage-table.txt"

Start at `0.5` and check ESS before lowering.

## Which group to reweight

--8<-- "snippets/reweight-table.txt"

Each *active* group is normalized to sum to its size; inactive groups get `1` (ESS stays `n`).

## When to skip

Stay unweighted when overlap is already good, you lack a reliable domain classifier, or you want the first-pass answer. Weights are for known overlap issues, not a default.

| Situation | Do |
|-----------|-----|
| No overlap concern | omit `weights` |
| Source outliers | `reweight="source"` |
| Target outliers | `reweight="target"` |
| Both | `reweight="both"` |
| Unsure | start unweighted, then compare |

Strings and `ss.ReweightMode` are interchangeable.

## References

- Shimodaira (2000). Covariate shift weighting. *JSPI*.
- Yamada et al. (2013). Relative density-ratio estimation. *Neural Computation*.
- Kamulete et al. (2022). Harmful shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
