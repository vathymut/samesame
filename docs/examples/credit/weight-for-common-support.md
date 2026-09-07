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
`deployment_risk`. Use them to calibrate the weights; predicted risk remains
the score tested for harmful movement.

## Compare the three weighting methods

Start with the unweighted result from the credit example, then compare all three
weighting methods for both probability models. Keep `worse` and the random seed
fixed. The code uses `shrinkage=0.5` for each method.

```python
unweighted = ss.test_harmful_shift(
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
        weighted_results[method] = ss.test_harmful_shift(
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
the less extreme p-value because the target weights are highly concentrated.
After excluding the split feature, the common-support comparison has more
effective target information and a smaller p-value.

## Check effective sample size

P-values do not show whether a few observations dominate the weighted result.
Check the effective sample size (ESS) for each weighting policy:

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
the split feature leaves only `23/2188` effective target observations, whereas
excluding it raises target ESS to `856/2188`. That is the key diagnostic for
choosing which domain model to report.

## Choose the method for this comparison

The HELOC split creates a low-risk source and a higher-risk target, so neither
group should automatically be treated as the clean reference population. The
all-feature domain model uses the splitting variable directly and therefore
creates an extremely narrow estimated common-support region. Excluding that
feature still gives strong domain discrimination but produces a less
concentrated comparison. Report both specifications when assessing sensitivity
to the split variable, and prefer the excluded-feature result when the goal is
to avoid encoding the split rule directly in the weights.

## Interpret the comparison

- A weaker weighted result would mean that the unweighted evidence was driven
  by low-overlap regions.
- A persistent result in both specifications means harmful shift remains after
  restricting the comparison toward common support.
- A low ESS means the weighted estimate is concentrated, not that the p-value
  is automatically invalid. Report ESS alongside the test result.

For the weighting API, see [Importance weights](../../api/weighting.md). For
the statistic being tested, see [How the harm test works](../../explanation/harmful-shift-statistic.md).
