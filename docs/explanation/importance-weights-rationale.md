# When importance weights help

Weighting helps when source and target differ for two reasons at once:

- there is a real change in the region you care about
- one or both groups also contain observations that sit where the other group almost never goes

Without weighting, those low-overlap observations can dominate the comparison. Weighting narrows it to **common support**.

For code, see [Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).

## The basic problem

A domain classifier estimates `P(target | x)`, call it `p̂(x)`. The standard density-ratio correction is:

$$
\hat{r}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

`n_source / n_target` is the prior ratio — inferred from group sizes, not a tunable parameter. It converts the classifier's posterior odds to a density ratio.

Plain `r̂` is powerful but unstable: when groups separate well, a few points receive huge weights and dominate the test.

--8<-- "snippets/clipping-note.txt"

## How `samesame` stabilises the weights

`samesame` uses relative importance weighting (RIW, Yamada et al. 2013), which blends the plain ratio toward uniform.

Source weighting:

$$
w_{\text{source}}(x) = \frac{\hat{r}(x)}{(1 - \lambda) + \lambda \hat{r}(x)}
$$

Target weighting:

$$
w_{\text{target}}(x) = \frac{1}{\lambda + (1 - \lambda) \hat{r}(x)}
$$

where `λ = shrinkage` in `[0, 1]`. You don't compute these by hand — `ss.domain_weights` does.

## What `shrinkage` (λ) does

| `shrinkage` | Effect |
|-------------|--------|
| `0.0` | Plain density ratio. Strongest correction, highest variance. |
| `0.5` | Default. Good balance. |
| `1.0` | Uniform weights. No correction. |

Lower = more aggressive; higher = more conservative. Start at `0.5` and check [ESS](../examples/weighting/diagnose-weight-concentration.md) before lowering.

## Choosing what to reweight

| Mode | What it does | Use when |
|------|--------------|----------|
| `reweight="source"` (`ss.ReweightMode.SOURCE`) | reweights source toward target; target unchanged | source has observations outside target support |
| `reweight="target"` (`ss.ReweightMode.TARGET`) | reweights target toward source; source unchanged | target has observations outside source support |
| `reweight="both"` (`ss.ReweightMode.BOTH`, default) | reweights both toward mutual support | both groups have low-overlap regions |

In all cases, `ss.domain_weights` normalizes each *active* group so weights sum to that group's size. Inactive groups get `1` for every observation (normalized to `n`, so uniform).

## When to skip weighting

Stay unweighted when:

- source and target already overlap well
- you don't have a reliable domain classifier
- you want the first-pass answer before narrowing to common support

Weights are for known overlap issues, not a default.

??? tip "Quick decision tree"
    *No overlap concern → omit `weights`.* *Source outliers only → `reweight="source"`.* *Both sides have outliers → `reweight="both"`.* *Unsure → start unweighted, then compare weighted.*

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative density-ratio estimation for robust distribution comparison. *Neural Computation*, 25(5), 1324–1370.
- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990) — the method behind `test_harmful_shift`.
