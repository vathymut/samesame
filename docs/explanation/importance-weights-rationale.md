# When importance weights help

Importance weights help when source and target differ for two reasons at once:

- there is a real change in the region you care about
- one or both groups also contain observations that sit in parts of feature space the other group
  almost never visits

Without weighting, those low-overlap observations can dominate the comparison.

For code, see
[Focus on shared support with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).

## The basic problem

Suppose a domain classifier estimates the probability that an observation belongs to target.
Call that probability $\hat{p}(x)$.

The standard density-ratio correction is:

$$
\hat{r}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

This is useful, but it can become unstable. When the groups are easy to separate, a small number
of observations can receive very large weights and dominate the test.

## How `samesame` stabilises the weights

`samesame` uses relative importance weighting (RIW), which blends the plain density ratio toward
uniform weighting.

For source weighting:

$$
w_{\text{source}}(x) = \frac{\hat{r}(x)}{(1 - \lambda) + \lambda \hat{r}(x)}
$$

For target weighting:

$$
w_{\text{target}}(x) = \frac{1}{\lambda + (1 - \lambda) \hat{r}(x)}
$$

You do not need to compute these by hand. `domain_weights(...)` does it for you.

## What `shrinkage` changes

| `shrinkage` | Effect |
|-----------|--------|
| `0.0` | Plain density ratio. Strongest correction, highest variance. |
| `0.5` | Practical default. Good balance between correction and stability. |
| `1.0` | Uniform weights. No correction. |

Lower values correct more aggressively. Higher values are more conservative.

## Choosing a mode

| Mode | Use it when |
|------|-------------|
| `reweight="source"` | source contains observations outside the target's common support |
| `reweight="target"` | target contains observations outside the source's common support |
| `reweight="both"` | both groups contain low-overlap observations and you want to focus on common support only |

In all three cases, `domain_weights(...)` normalizes each active group so the weights
sum to that group's sample size.

## When to skip weighting

Start unweighted when:

- source and target already overlap well
- you do not have a reliable domain classifier
- you want the first-pass answer before narrowing attention to common support

Weights are most useful when you already know that overlap is the issue, not as a default for every
comparison.

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the
  log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227-244.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative
  density-ratio estimation for robust distribution comparison. *Neural Computation*, 25(5),
  1324-1370.
