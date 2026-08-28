# Reweight

Policy for `domain_weights(..., reweight=...)` — which group(s) to adjust toward common support. String or `ss.ReweightMode` — interchangeable, enum gives autocomplete.

| Mode | What it does | Use when |
|------|--------------|----------|
| `reweight="source"` (`ss.ReweightMode.SOURCE`) | reweights source toward target; target unchanged | source has low-overlap points outside target support |
| `reweight="target"` (`ss.ReweightMode.TARGET`) | reweights target toward source; source unchanged | target has low-overlap points outside source support |
| `reweight="both"` (`ss.ReweightMode.BOTH`, default) | reweights both toward mutual support | both groups have low-overlap regions |

Inactive groups get weight `1` (uniform after normalization; ESS stays `n`).

**Guidance:** start unweighted, then compare weighted. See quick decision in [When importance weights help](../importance-weights-rationale.md) and the three-view comparison in [Restrict to common support on both sides](../../examples/weighting/double-weighting.md).

See also: [Common support](common-support.md), [Shrinkage](shrinkage.md), [Importance weights](importance-weights.md).
