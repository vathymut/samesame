# How to: Focus harmful-shift testing on common support

Use this guide when source contains observations that deployment will rarely or never see and you want the test to focus on comparable cases.

This is source-side reweighting: keep target unchanged, down-weight source points that look unlikely under the target distribution.

## Step 1 — Recreate the baseline workflow

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

`domain_prob` (`P(target|x)`) is for weighting only. The harmful-shift signal is still predicted default risk. Do not reuse `domain_prob` as the harm score.

--8<-- "snippets/honest-scores.txt"

## Step 2 — Build source-side importance weights

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

print(f"Unweighted p-value: {unweighted.pvalue:.4f}")
print(f"Weighted   p-value: {weighted.pvalue:.4f}")
```

--8<-- "snippets/clipping-note.txt"

## Step 3 — Interpret the difference

- **Unweighted** — every observation at full strength.
- **Weighted** — source points unlikely under target are down-weighted (emphasizes common support from the source side).
- If the result weakens substantially after weighting, the original signal was driven by parts of training not representative of deployment.
- If the result stays strong, the shift persists on common support.

Use this mode when training contains outliers or edge cases not representative of the population you now care about.

If deployment also contains low-overlap cases, continue to [Restrict testing to common support on both sides](double-weighting.md).

??? tip "Check weight concentration"
    After building weights, call `weights.effective_sample_size()` to ensure the correction is not driven by a handful of points. See [Diagnose weight concentration](diagnose-weight-concentration.md).
