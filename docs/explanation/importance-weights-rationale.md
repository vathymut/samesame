# Why importance weights stabilise shift detection

This page explains the conceptual landscape behind importance weighting in \`samesame\`.
It covers why plain density-ratio weights can become extreme, what \`alpha_blend\` trades off,
and when each of the three weighting modes is appropriate.

There are no code examples here. For worked code, see the tutorial
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).

---

## The density-ratio problem

A classifier two-sample test trains a classifier to distinguish source from target samples and
uses its predicted probabilities as shift scores. When covariate shift is present — source and
target differ in their feature distributions — the ideal correction is to reweight source samples
by the **density ratio**:

$$
w(x) = \frac{p_{\text{target}}(x)}{p_{\text{source}}(x)}
$$

In practice, this ratio is estimated from a membership classifier. Let $\hat{p}(x)$ be the
predicted probability that sample $x$ belongs to the target group. Then the estimated density
ratio is:

$$
\hat{w}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)} \cdot \frac{n_{\text{source}}}{n_{\text{target}}}
$$

The problem is that when the classifier separates groups well — exactly the situation where you
need weighting — the estimated $\hat{p}(x)$ for source samples in the overlap region can
approach 0.5 while for isolated outliers it approaches 0. Division by a near-zero denominator
then produces extreme weight values. One or a handful of source outliers can dominate the entire
weighted test, masking real signal in the shared region.

---

## Taming the ratio with alpha blending

\`samesame\` uses **Relative Importance Weighting (RIW)** to stabilise the density ratio.
The RIW weight for a source sample is:

$$
w_{\text{source}}(x) = \frac{\hat{w}(x)}{(1 - \alpha) + \alpha \cdot \hat{w}(x)}
$$

where $\alpha$ is \`alpha_blend\`. For target samples receiving inverse weights:

$$
w_{\text{target}}(x) = \frac{1}{\alpha + (1 - \alpha) \cdot \hat{w}(x)}
$$

The blended denominator in both formulas prevents any single weight from growing without bound.
The parameter \`alpha_blend\` controls the trade-off:

| \`alpha_blend\` | Effect |
|----------------|--------|
| \`0.0\` | Plain density ratio; equivalent to IWERM. Maximum variance. |
| \`0.5\` (default) | Balanced blend; practical default for most applications. |
| \`1.0\` | All weights become uniform (no correction at all). |

Lower values apply more aggressive correction but increase variance. Higher values are more
conservative. The default \`alpha_blend=0.5\` is a good starting point; reduce it if you are
confident the overlap region is large and the classifier is well-calibrated.

---

## Three weighting modes

\`contextual_weights\` and the \`membership_prob\` path in \`test_shift\` / \`test_adverse_shift\`
support three modes that differ in which group receives non-unit weights:

### source — source reweighting

Source samples that look unlike any target sample receive downweighted importance.
Target samples all receive weight 1. Use this when the source group contains outliers
that are foreign to the target population and you want the test to focus on the overlap region
from the source side.

### target — target reweighting

Target samples that look unlike any source sample receive downweighted importance.
Source samples all receive weight 1. Use this when the target group contains outliers
foreign to the source population.

### both — double-weighting

Both source and target outliers are downweighted simultaneously. Both sides of the density
ratio are corrected. Use this when both groups contain outliers foreign to the other group
and you want the test to focus exclusively on the common support region. This is the most
aggressive correction and should be chosen only when outliers are genuinely present in both
groups.

---

## The group balance correction

When source and target group sizes differ, the raw density ratio is biased. By default
(\`balance=True\`), \`contextual_weights\` corrects for this automatically by multiplying the
density ratio by $n_{\text{source}} / n_{\text{target}}$. This is equivalent to specifying
the prior ratio as the observed class balance.

Set \`balance=False\` only if your membership classifier was already trained with equal class
weights, or if you have an external reason to assume equal group sizes.

---

## Decision guide

| Scenario | Recommended mode | Key parameter |
|----------|-----------------|---------------|
| First-pass check; source contains outliers foreign to target | \`mode="source"\` with \`alpha_blend=0.5\` | \`alpha_blend\` |
| Target contains outliers foreign to source | \`mode="target"\` with \`alpha_blend=0.5\` | \`alpha_blend\` |
| Both groups contain outliers foreign to the other | \`mode="both"\` with \`alpha_blend=0.5\` | \`mode\`, \`alpha_blend\` |

---

## References

- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the
  log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2), 227–244.
- Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., & Sugiyama, M. (2013). Relative
  density-ratio estimation for robust distribution comparison. *Neural Computation*, 25(5),
  1324–1370.
