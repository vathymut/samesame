# How to: Restrict testing to common support on both sides

Use this guide when both source and target contain low-overlap observations and source-only reweighting is not enough.

This mode reweights **both** groups and can substantially change the comparison. Use it only when low-overlap observations occur in both groups.

## Step 1 — Start from the source-reweighting setup

This guide continues from [Focus harmful-shift testing on common support](source-reweighting.md). At that point you already have:

- `source_prob` and `target_prob` — domain probabilities `P(target | x)` from the domain classifier
- `train_risk` and `deployment_risk` — the harmful-shift signal (predicted risk)

See that page for the full setup snippet — the code below continues from it.

## Step 2 — Weight both groups

```python
import samesame as ss

weights_both = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="both",  # or ss.ReweightMode.BOTH
    shrinkage=0.5,
)

double_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights_both,
    rng=np.random.default_rng(12345),
)

print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

--8<-- "snippets/clipping-note.txt"

## Step 3 — Compare the three views

Continuing from Step 2 (and [source reweighting](source-reweighting.md) for `unweighted`):

```python
# rebuild unweighted here if you didn't run the previous page
unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)

weights_source = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",
    shrinkage=0.5,
)

source_weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights_source,
    rng=np.random.default_rng(12345),
)

print(f"Unweighted      p-value: {unweighted.pvalue:.4f}")
print(f"Source-weighted p-value: {source_weighted.pvalue:.4f}")
print(f"Doubly-weighted p-value: {double_weighted.pvalue:.4f}")
```

Think of the three results this way:

- **Unweighted** — full populations.
- **Source-weighted** — emphasizes overlap from the source side; target unchanged.
- **Doubly-weighted** — emphasizes common support from both sides.

If the signal shrinks only after doubly-weighting, target-side low-overlap points were still influencing the result after source reweighting.

## Choosing `shrinkage` (λ)

`shrinkage` trades correction strength against stability:

- `shrinkage=0.0` — plain density ratio. Strongest correction, highest variance.
- `shrinkage=0.5` — default. Good balance.
- `shrinkage=1.0` — uniform weights. No correction.

Lower values correct more aggressively. Start at `0.5`, inspect [effective sample size](diagnose-weight-concentration.md), and only lower it when domain probabilities and overlap are reliable.

For the formulas behind this, see [When importance weights help](../../explanation/importance-weights-rationale.md).

## Next steps

- Diagnose concentration with [Diagnose weight concentration](diagnose-weight-concentration.md) — regroup when `ESS < n/4`.
- For the API, see [Importance weights](../../api/weighting.md).
