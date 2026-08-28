# How to: Restrict testing to common support on both sides

Use this when both groups have low-overlap regions and source-only reweighting isn't enough. This reweights **both** groups and can substantially change the comparison — use only when low overlap occurs on both sides.

See [Glossary: Reweight](../../explanation/glossary.md#reweight).

!!! info "Prerequisites"
    - [Focus harmful-shift testing on common support](source-reweighting.md) — source-only weighting setup reused here.
    - [When importance weights help](../../explanation/importance-weights-rationale.md) — `shrinkage` and ESS.

## Step 1 — Start from the source-reweighting setup

This guide continues from [Focus harmful-shift testing on common support](source-reweighting.md). At that point you have:

- `source_prob` and `target_prob` — domain probabilities `P(target | x)` from the domain classifier
- `train_risk` and `deployment_risk` — the harm signal (predicted risk)

See that page for the full setup — the code below continues from it.

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
# rebuild unweighted if you didn't run the previous page
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

Three views:

- **Unweighted** — full populations.
- **Source-weighted** — emphasizes overlap from the source side; target unchanged.
- **Doubly-weighted** — emphasizes common support from both sides.

If the signal shrinks only after doubly-weighting, target-side low-overlap points were still influencing the result.

## Choosing `shrinkage` (λ)

--8<-- "snippets/shrinkage-table.txt"

Inspect ESS before lowering — see [Diagnose weight concentration](diagnose-weight-concentration.md). For formulas, see [When importance weights help](../../explanation/importance-weights-rationale.md).

??? example "Full script — copy and run (continues from source-reweighting setup)"
    ```python
    --8<-- "examples/weighting/_code/double_weighting_example.py:full"
    ```

## Next steps

- Diagnose concentration with [Diagnose weight concentration](diagnose-weight-concentration.md):

--8<-- "snippets/ess-rule.txt"

- For the API, see [Importance weights](../../api/weighting.md).
