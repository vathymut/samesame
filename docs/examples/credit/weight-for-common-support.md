# Weight for common support

This tutorial continues [Monitor a credit model](monitor-credit.md). Its
unweighted test found evidence of a shift toward higher predicted risk. It
gives every application equal weight, including observations in regions with
little overlap.

**Common support** is the part of feature space represented by both groups.
Here we ask whether the harmful shift remains when we give less weight to
observations in regions poorly represented by the other group.

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

The domain probability estimates how likely each application is to belong to
the target group. Fit the classifier on the same source-target split used by
the credit example. Calibration aligns predicted probabilities with observed
frequencies; `CalibratedClassifierCV` applies it to the random forest. The
outer `cross_val_predict` call produces each row's probability using a model
that did not train on that row.

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

The table compares the two domain models. ROC AUC (area under the receiver
operating characteristic curve) shows how well each separates source from
target: `0.5` is chance; values farther from `0.5` mean stronger separation.
Effective sample size (ESS) summarizes weight concentration. Compare it with
the group's sample size: a lower `ESS/n` means a few observations carry more
weight.

| Metric | All features | Excluding split feature |
|---|---:|---:|
| Domain AUC | 1.000 | 0.951 |
| Mean source probability | 0.001 | 0.094 |
| Mean target probability | 0.996 | 0.673 |
| Common-support source ESS | 2,530 | 2,988 |
| Common-support target ESS | 23 | 856 |
| Common-support p-value | 0.0019 | 0.0001 |

Including the split feature produces almost perfect domain separation and very
extreme probabilities. Excluding it produces more overlap and raises the
target's effective sample size. The classifier still separates the groups
well.

Keep these domain probabilities separate from `train_risk` and
`deployment_risk`. Use them to build the weights. Predicted risk remains
the score tested for harmful movement.

## Compare the three weighting methods

Start with the unweighted result from the credit example. Compare all three
weighting methods for both probability models, with `worse` and the random seed
fixed. The code uses `shrinkage=0.5`, which blends raw importance weights
(density ratios) toward uniform weights. `0` uses the raw density ratio; `1`
gives uniform weights, so there is no correction. Check effective sample size
before lowering shrinkage.

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

The harmful shift remains detectable under all three methods and both
domain-model specifications. With all features, common-support weighting gives
a less extreme p-value and highly concentrated target weights. Excluding the
split feature raises target ESS and lowers the p-value. The smaller p-value
does not, by itself, mean the shift is larger.

## Check effective sample size

P-values do not show whether a few observations dominate the weighted result.
Compare each group's ESS with its sample size for every weighting policy:

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

Source-weighting leaves target ESS at `n`; target-weighting leaves source ESS
at `n`; common-support weighting reduces ESS in both groups. The target ESS is
`23/2188` with the split feature and `856/2188` without it, showing how strongly
the feature concentrates target weights. Report ESS alongside each result.

## Choose the method for this comparison

The HELOC split creates a low-risk source and a higher-risk target. Neither
group is automatically the right reference for an overlap-adjusted comparison.
The all-feature domain model uses the split variable directly, so it estimates
an extremely narrow common-support region. Excluding that feature still
separates the groups but produces less concentrated weights. Report both
specifications to show sensitivity to the split variable. If you want weights
independent of the split rule, prefer the excluded-feature model.

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
