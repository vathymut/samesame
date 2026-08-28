# When importance weights help

Weighting helps when source and target differ for two reasons at once:

- there is a real change in the region you care about
- one or both groups also contain points where the other group almost never goes (low overlap)

Without weighting, those low-overlap points can dominate a permutation test even though they tell you little about the shared population. Weighting reframes the comparison to **common support**.

> **Example:** source (training) has many 20-year-old students that production never sees, while production has retirees never seen in training. An unweighted test is swayed by those extremes; a weighted test focuses on the 30–60 overlap.

For code, see [Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md) and [Importance weights API](../api/weighting.md).

## The basic problem

A domain classifier estimates `P(target | x)`, call it `p̂(x)`. The standard density-ratio correction is:

$$
\hat{r}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

`n_source / n_target` is the prior ratio — inferred from group sizes, not a tunable parameter. It converts posterior odds to a density ratio.

Plain `r̂` is powerful but unstable: when groups separate well, a few points receive huge weights and dominate the test. See [Diagnose weight concentration](../examples/weighting/diagnose-weight-concentration.md).

--8<-- "snippets/clipping-note.txt"

## How `samesame` stabilises the weights

`samesame` uses relative importance weighting (RIW, Yamada et al. 2013) — a blend of the plain ratio toward uniform. You don't compute these by hand; `ss.domain_weights` does. In short: lower `shrinkage` is stronger correction, higher is more stable; `0.5` is the balanced default.

??? example "RIW formulas (for experts)"

    With $\hat{r}(x)$ from above and $\lambda = $ `shrinkage` in $[0,1]$:

    $$
    w_{\text{source}}(x) = \frac{\hat{r}(x)}{(1 - \lambda) + \lambda \hat{r}(x)}
    $$

    $$
    w_{\text{target}}(x) = \frac{1}{\lambda + (1 - \lambda) \hat{r}(x)}
    $$

## What `shrinkage` (λ) does

--8<-- "snippets/shrinkage-table.txt"

Lower = stronger correction but higher variance. Start at `0.5` and check ESS before lowering — see [Diagnose weight concentration](../examples/weighting/diagnose-weight-concentration.md) and [Glossary: Shrinkage](glossary.md#shrinkage).

## Choosing what to reweight

--8<-- "snippets/reweight-table.txt"

In all cases `ss.domain_weights` normalizes each *active* group so weights sum to that group's size. Inactive groups get `1` for every observation (uniform after normalization; ESS stays `n`).

## When to skip weighting

Stay unweighted when:

- source and target already overlap well
- you don't have a reliable domain classifier
- you want the first-pass answer before narrowing to common support

Weights are for known overlap issues, not a default.

**Quick decision:**

- No overlap concern → omit `weights`.
- Source outliers only → `reweight="source"`.
- Target outliers only → `reweight="target"`.
- Both sides have outliers → `reweight="both"` for common-support comparison.
- Unsure → start unweighted, then compare weighted.

Strings and `ss.ReweightMode` enum are interchangeable (`"source"` ↔ `ss.ReweightMode.SOURCE`).

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative density-ratio estimation for robust distribution comparison. *Neural Computation*, 25(5), 1324–1370.
- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990) — the method behind `test_harmful_shift`.
