# How to: Restrict testing to common support on both sides

Use this guide when both source and target contain low-overlap observations and source-only
reweighting is not enough.

This mode applies weighting to both groups and can substantially change the comparison. Use it when
low-overlap observations occur in both groups.

## Step 1 - Start from the source-reweighting setup

This guide continues from
[Focus harmful-shift testing on shared support](source-reweighting.md).
At that point you already have:

- `source_prob` and `target_prob` from the domain classifier
- `train_risk` and `deployment_risk` as the harmful-shift signal

## Step 2 - Weight both groups

```python
from samesame.weights import from_domain_probabilities

weights_both = from_domain_probabilities(
  source_prob=source_prob,
  target_prob=target_prob,
  mode="both",
  lambda_=0.5,
)

double_weighted = ss.detect_harm(
  source=train_risk,
  target=deployment_risk,
  direction=ss.Direction.HIGHER_IS_WORSE,
  weights=weights_both,
  random_state=12345,
)

print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

## Step 3 - Compare the three views

```python
weights_source = from_domain_probabilities(
  source_prob=source_prob,
  target_prob=target_prob,
  mode="source",
  lambda_=0.5,
)

source_weighted = ss.detect_harm(
  source=train_risk,
  target=deployment_risk,
  direction=ss.Direction.HIGHER_IS_WORSE,
  weights=weights_source,
  random_state=12345,
)

print(f"Unweighted      p-value: {unweighted.pvalue:.4f}")
print(f"Source-weighted p-value: {source_weighted.pvalue:.4f}")
print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

Think of the three results this way:

- **Unweighted** looks at the full populations.
- **Source-weighted** focuses on overlap from the source side.
- **Doubly-weighted** focuses on common support from both sides.

If the signal shrinks only after doubly-weighting, target-side outliers were still influencing the
result after source reweighting.

## Choosing `lambda_`

`lambda_=0.5` is a practical starting point.

- Lower values make the correction stronger and the variance higher.
- Higher values produce weights closer to uniform.

If you are unsure, start at `0.5`, inspect the sensitivity, and only choose a lower value when the
domain probabilities and assumed overlap structure are reliable.

For the intuition behind the formulas, see
[When importance weights help](../../explanation/importance-weights-rationale.md).
