# Weight for common support

This tutorial extends [Monitor a credit model](monitor-credit.md). The earlier
example found a harmful shift toward higher predicted risk between the HELOC
source and target groups. Here we ask whether that result remains after
reducing the influence of observations outside their common support.

The three weighting methods answer different comparison questions:

| Method | What it changes | When to use it |
|--------|-----------------|----------------|
| **Source-weighted** | Reweights source toward target; target stays unchanged. | Only source has important low-overlap observations. |
| **Target-weighted** | Reweights target toward source; source stays unchanged. | Only target has important low-overlap observations. |
| **Common-support** | Reweights both groups toward their mutual support. | Both groups may contain outliers or low-overlap observations. |

## Prerequisites

Complete [Monitor a credit model](monitor-credit.md) first. This tutorial uses
the same HELOC split, source-trained risk model, and variables:

- `X_train`, `X_deployment`: source and target features;
- `train_risk`, `deployment_risk`: source and target predicted-risk scores; and
- `ss`: the `samesame` package imported as `import samesame as ss`.
- `np`: NumPy imported as `import numpy as np`.

The complete standalone script is
[`common_support_heloc.py`](https://github.com/vathymut/samesame/blob/main/docs/examples/weighting/_code/common_support_heloc.py).

## Estimate domain probabilities

The domain probability estimates how target-like each application is. Fit the
domain classifier on the same source-target split used by the credit example.
`CalibratedClassifierCV` calibrates the random-forest probabilities, while the
outer `cross_val_predict` call keeps each row's probability out of sample.

```python
--8<-- "snippets/heloc-split.py:heloc-domain"
source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]
```

Keep these domain probabilities separate from `train_risk` and
`deployment_risk`. Use them to calibrate the weights; predicted risk remains
the score tested for harmful movement.

## Compare the three weighting methods

Start with the unweighted result from the credit example, then compare all three
weighting methods. Keep `worse` and the random seed fixed. The code uses
`shrinkage=0.5` for each method.

```python
unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)

w_source = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",
    shrinkage=0.5,
)
w_target = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="target",
    shrinkage=0.5,
)
w_common_support = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="both",
    shrinkage=0.5,
)

source_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=w_source,
    rng=np.random.default_rng(12345),
)
target_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=w_target,
    rng=np.random.default_rng(12345),
)
common_support = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=w_common_support,
    rng=np.random.default_rng(12345),
)

print(f"Unweighted      p-value: {unweighted.pvalue:.4f}")
print(f"Source-weighted p-value: {source_weighted.pvalue:.4f}")
print(f"Target-weighted p-value: {target_weighted.pvalue:.4f}")
print(f"Common-support p-value:  {common_support.pvalue:.4f}")
```

Expected output:

```text
Unweighted      p-value: 0.0001
Source-weighted p-value: 0.0001
Target-weighted p-value: 0.0019
Common-support p-value:  0.0019
```

The harmful shift remains detectable under all three weighting methods. The
target-weighted and common-support results agree here because the target side
contains the most concentrated low-overlap observations.

## Check effective sample size

P-values do not show whether a few observations dominate the weighted result.
Check the effective sample size (ESS) for each weighting policy:

```python
for label, weights in [
    ("source-weighted", w_source),
    ("target-weighted", w_target),
    ("common-support", w_common_support),
]:
    ess = weights.effective_sample_size()
    print(
        f"{label}: ESS source {ess.source:.0f}/{len(source_prob)}, "
        f"target {ess.target:.0f}/{len(target_prob)}"
    )
```

Expected output:

```text
source-weighted: ESS source 2530/7683, target 2188/2188
target-weighted: ESS source 7683/7683, target 23/2188
common-support: ESS source 2530/7683, target 23/2188
```

Source-weighted leaves the target ESS at `n`, while target-weighted leaves the
source ESS at `n`. Common-support weighting reduces ESS on both sides. Here the
source ESS is about `0.33n` and the target ESS is only `23/2188` for the
target-weighted and common-support methods. That is the key diagnostic: the
common-support comparison is based on very little effective target information.

## Choose the method for this comparison

The HELOC split creates a low-risk source and a higher-risk target, so neither
group should automatically be treated as the clean reference population. The
ESS results show that the target has a particularly small common-support
region, but the source also loses substantial effective sample size when it is
reweighted. Because both groups may contain observations outside the other's
support, **common-support** is the appropriate method for the primary
overlap-adjusted comparison. Report its low target ESS as a limitation rather
than presenting the weighted p-value without context.

## Interpret the comparison

- A weaker weighted result would mean that the unweighted evidence was driven
  by low-overlap regions.
- A persistent result, as in this example, means harmful shift remains after
  restricting the comparison toward common support.
- A low ESS means the weighted estimate is concentrated, not that the p-value
  is automatically invalid. Report ESS alongside the test result.

For the weighting API, see [Importance weights](../../api/weighting.md). For
the statistic being tested, see [How the harm test works](../../explanation/harmful-shift-statistic.md).
