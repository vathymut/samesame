# Weight for common support

This tutorial continues [Monitor a credit model](monitor-credit.md). That
example found a harmful shift toward higher predicted risk. It counted
every application, including observations in regions with little overlap.

**Common support** is the part of feature space where observations from
both groups can occur. Here we ask whether the harmful shift remains after
down-weighting observations outside that region.

The three methods change who gets down-weighted:

| Method | What it changes | When to use it |
|--------|-----------------|----------------|
| **Source-weighted** | Reweights source toward target; target stays unchanged. | Source has observations in regions with little target overlap. |
| **Target-weighted** | Reweights target toward source; source stays unchanged. | Target has observations in regions with little source overlap. |
| **Common-support** | Reweights both groups toward their mutual support. | Both groups have observations in regions with little overlap. |

## Prerequisites

Complete [Monitor a credit model](monitor-credit.md) first. This tutorial uses
the same HELOC split, source-trained risk model, and variables:

- `X_train`, `X_deployment`: source and target features
- `train_risk`, `deployment_risk`: source and target predicted-risk scores
- `ss`: `samesame`, imported as `import samesame as ss`
- `np`: NumPy, imported as `import numpy as np`

The complete standalone script is
[`common_support_heloc.py`](https://github.com/vathymut/samesame/blob/main/docs/examples/weighting/_code/common_support_heloc.py).

## Estimate domain probabilities

The domain probability estimates how target-like each application is. Fit the
domain classifier on the same source-target split used by the credit example.
`CalibratedClassifierCV` calibrates the random-forest probabilities, while the
outer `cross_val_predict` call keeps each row's probability out of sample.

Because `ExternalRiskEstimate` defines the split, estimate probabilities both
with all features and with that split feature excluded. The first version shows
the result when the domain model can use the splitting rule directly; the
second asks how much separation the remaining features support.

```python
--8<-- "snippets/heloc-split.py:heloc-domain"
probability_sets = {
    label: (
        probabilities[split.values == 0],
        probabilities[split.values == 1],
    )
    for label, probabilities in domain_probabilities.items()
}
```

The two probability models differ substantially:

| Metric | All features | Excluding split feature |
|---|---:|---:|
| Domain AUC | 1.000 | 0.951 |
| Mean source probability | 0.001 | 0.094 |
| Mean target probability | 0.996 | 0.673 |
| Common-support source ESS | 2,530 | 2,988 |
| Common-support target ESS | 23 | 856 |
| Common-support p-value | 0.0019 | 0.0001 |

Including the split feature produces almost perfect domain separation and very
extreme probabilities. Excluding it produces more overlap and substantially
more effective target information after weighting, while the classifier still
distinguishes the groups well.

Keep these domain probabilities separate from `train_risk` and
`deployment_risk`. Use them to build the weights. Predicted risk remains
the score tested for harmful movement.

## Compare the three weighting methods

Start with the unweighted result from the credit example, then compare all three
weighting methods for both probability models. Keep `worse` and the random seed
fixed. The code uses `shrinkage=0.5`, which blends density-ratio weights
toward uniform weights. `0` is the raw density ratio; `1` is uniform, so
there is no correction. Check effective sample size before lowering the
shrinkage.

```python
unweighted = ss.test_harm(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)
print(f"Unweighted p-value: {unweighted.pvalue:.4f}")

for label, (source_prob, target_prob) in probability_sets.items():
    print(f"\n{label}")
    weighted_results = {}
    for method, reweight in [
        ("Source-weighted", "source"),
        ("Target-weighted", "target"),
        ("Common-support", "both"),
    ]:
        weights = ss.domain_weights(
            source=source_prob,
            target=target_prob,
            reweight=reweight,
            shrinkage=0.5,
        )
        weighted_results[method] = ss.test_harm(
            source=train_risk,
            target=deployment_risk,
            worse="higher",
            weights=weights,
            rng=np.random.default_rng(12345),
        )
    for method, result in weighted_results.items():
        print(f"{method}: p-value {result.pvalue:.4f}")
```

Expected output:

```text
Unweighted p-value: 0.0001

all features
Source-weighted: p-value 0.0001
Target-weighted: p-value 0.0001
Common-support: p-value 0.0019

excluding split feature
Source-weighted: p-value 0.0001
Target-weighted: p-value 0.0001
Common-support: p-value 0.0001
```

The harmful shift remains detectable under all three weighting methods and both
domain-model specifications. With all features, common-support weighting gives
a less extreme p-value alongside highly concentrated target weights. Excluding
the split feature gives a larger target effective sample size and a smaller
p-value; the smaller p-value does not, by itself, mean the shift is larger.

## Check effective sample size

P-values do not show whether a few observations dominate the weighted result.
Check the effective sample size (ESS), which summarizes how concentrated the
weights are, for each weighting policy:

```python
for label, (source_prob, target_prob) in probability_sets.items():
    print(label)
    for method, reweight in [
        ("source-weighted", "source"),
        ("target-weighted", "target"),
        ("common-support", "both"),
    ]:
        weights = ss.domain_weights(
            source=source_prob,
            target=target_prob,
            reweight=reweight,
            shrinkage=0.5,
        )
        ess = weights.effective_sample_size()
        print(
            f"{method}: ESS source {ess.source:.0f}/{len(source_prob)}, "
            f"target {ess.target:.0f}/{len(target_prob)}"
        )
```

Expected output:

```text
all features
source-weighted: ESS source 2530/7683, target 2188/2188
target-weighted: ESS source 7683/7683, target 23/2188
common-support: ESS source 2530/7683, target 23/2188

excluding split feature
source-weighted: ESS source 2988/7683, target 2188/2188
target-weighted: ESS source 7683/7683, target 856/2188
common-support: ESS source 2988/7683, target 856/2188
```

Source-weighted leaves the target ESS at `n`, while target-weighted leaves the
source ESS at `n`. Common-support weighting reduces ESS on both sides. Including
the split feature drops target ESS to only `23/2188`; excluding it raises
target ESS to `856/2188`. This shows how strongly the split feature concentrates the
target weights. Report ESS alongside each result.

## Choose the method for this comparison

The HELOC split creates low-risk source and higher-risk target groups. Neither
is automatically the clean reference for an overlap-adjusted comparison. The
all-feature domain model uses the split variable directly, which makes the
estimated common-support region extremely narrow. Excluding that feature still
separates the groups but produces less concentrated weights. Report both
specifications to show sensitivity to the split variable. Prefer the
excluded-feature result if you want to avoid using the split rule directly to
build the weights.

## Interpret the comparison

- If evidence weakens after weighting, some of the unweighted signal may come
  from low-overlap regions. Interpret the result alongside ESS.
- If the shift remains detectable in both specifications, there is evidence
  that it persists after restricting the comparison toward common support.
- A low ESS warns that a few observations carry much of the weight; it does not
  automatically invalidate the permutation p-value. Report ESS alongside the
  test result.

For the weighting API, see [Importance weights](../../api/weighting.md). For
the statistic being tested, see [How the harm test works](../../explanation/harmful-shift-statistic.md).
