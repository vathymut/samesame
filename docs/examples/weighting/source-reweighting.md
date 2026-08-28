# How to: Focus harmful-shift testing on common support

Down-weight source observations that rarely occur in target. Target unchanged; source points unlikely under target get low weight.

Use this when your source (training) contains cases the target (deployment) will rarely see and you want the test to focus on comparable cases. See [Glossary: Common support](../../explanation/glossary.md#common-support).

!!! info "Prerequisites"
    - [Adjust for covariate shift with importance weights](../tutorials/adjust-for-covariate-shift.md) — domain weights and `reweight`/`shrinkage`.
    - [Monitor predicted credit risk](../credit/monitor-credit-risk.md) — HELOC split used here.

## Step 1 — Baseline

This example uses the same HELOC split as [Monitor predicted credit risk](../credit/monitor-credit-risk.md).

```python
import numpy as np
import samesame as ss

--8<-- "snippets/heloc-split.py:heloc-split"
--8<-- "snippets/heloc-split.py:heloc-domain"
--8<-- "snippets/heloc-split.py:heloc-risk-model"

unweighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    rng=np.random.default_rng(12345),
)
```

`domain_prob` (`P(target | x)`) is for weighting only. The harm signal is still predicted default risk — don't reuse `domain_prob` as the harm score.

--8<-- "snippets/honest-scores-ref.txt"

## Step 2 — Build source-side weights

```python
source_prob = domain_prob[split.values == 0]
target_prob = domain_prob[split.values == 1]

weights = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="source",
    shrinkage=0.5,
)

weighted = ss.test_harmful_shift(
    source=train_risk,
    target=deployment_risk,
    worse="higher",
    weights=weights,
    rng=np.random.default_rng(12345),
)

print(f"Unweighted p-value: {unweighted.pvalue:.4f}")  # → 0.0001
print(f"Weighted   p-value: {weighted.pvalue:.4f}")    # → 0.0001 (persists on common support)
```

--8<-- "snippets/clipping-note.txt"

## Step 3 — Read the difference

--8<-- "snippets/pvalue-guidance.txt"

- **Unweighted** — every observation at full strength.
- **Weighted** — source points unlikely under target are down-weighted (common support from the source side).
- If the result weakens after weighting, the signal was driven by source regions not representative of target.
- If it stays strong, the shift persists on common support.

Use this mode when source contains outliers not representative of the population you now care about.

??? example "Full script — copy and run"
    ```python
    --8<-- "examples/weighting/_code/source_reweighting_example.py:full"
    ```

## Next steps

- If target also has low-overlap regions, see [Restrict testing to common support on both sides](double-weighting.md).
- Check concentration before trusting the weighted result:

--8<-- "snippets/ess-rule.txt"

See [Diagnose weight concentration](diagnose-weight-concentration.md).

??? tip "Check weight concentration"
    After building weights, call `weights.effective_sample_size()` — compare each ESS to its `n`.
