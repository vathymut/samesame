# When importance weights help

Importance weights help when source and target differ for two reasons at once:

- there is a real change in the region you care about
- one or both groups also contain observations that sit where the other group almost never goes

Without weighting, those low-overlap observations can dominate the comparison. Weighting narrows it to **common support**.

For code, see [Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).

## The basic problem

A domain classifier estimates `P(target | x)`, the domain probability. Call that `p̂(x)`. The standard density-ratio correction is:

$$
\hat{r}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

`n_source / n_target` is the prior ratio — inferred from group sizes, not a tunable `balance` parameter. This ratio converts the classifier's posterior odds to a density ratio.

Plain `r̂` is powerful but unstable: when groups separate well, a few points can receive huge weights and dominate the test.

--8<-- "snippets/clipping-note.txt"

## How `samesame` stabilises the weights

`samesame` uses relative importance weighting (RIW, Yamada et al. 2013), which blends the plain ratio toward uniform.

For source weighting:

$$
w_{\text{source}}(x) = \frac{\hat{r}(x)}{(1 - \lambda) + \lambda \hat{r}(x)}
$$

For target weighting:

$$
w_{\text{target}}(x) = \frac{1}{\lambda + (1 - \lambda) \hat{r}(x)}
$$

where `λ = shrinkage` in `[0, 1]`. You don't compute these by hand — `ss.domain_weights` does it.

## What `shrinkage` (λ) changes

| `shrinkage` | Effect |
|-------------|--------|
| `0.0` | Plain density ratio. Strongest correction, highest variance. |
| `0.5` | Default. Good balance between correction and stability. |
| `1.0` | Uniform weights. No correction. |

Lower = more aggressive; higher = more conservative. Start at `0.5` and check [ESS](../examples/weighting/diagnose-weight-concentration.md) before lowering.

## Choosing a mode

| Mode | Emphasizes | Use when |
|------|-----------|----------|
| `reweight="source"` (`ss.ReweightMode.SOURCE`) | overlap from the source side; target unchanged | source contains observations outside target's common support |
| `reweight="target"` (`ss.ReweightMode.TARGET`) | overlap from the target side; source unchanged | target contains observations outside source's common support |
| `reweight="both"` (`ss.ReweightMode.BOTH`, default) | common support from both sides | both groups contain low-overlap observations |

In all cases, `ss.domain_weights` normalizes each *active* group so weights sum to that group's sample size. Inactive groups get `1` for every observation (normalized to `n`, so they are uniform).

## When to skip weighting

Start *unweighted* when:

- source and target already overlap well
- you don't have a reliable domain classifier
- you want the first-pass answer before narrowing to common support

Weights are most useful when you already know overlap is the issue, not as a default.

??? tip "Quick decision tree"
    *No overlap concern → omit `weights`.* *Source outliers only → `reweight="source"`.* *Both sides have outliers → `reweight="both"`.* *Unsure → start unweighted, then compare weighted.*

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative density-ratio estimation for robust distribution comparison. *Neural Computation*, 25(5), 1324–1370.
- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990) — the method behind `test_harmful_shift`.
