# Effective sample size (ESS)

Kish's ESS per group — diagnostics for weight concentration.

```
ESS = (sum w)² / sum w²
```

- `ESS = n` for uniform weights.
- `ESS → 1` when one point dominates.

```python
ess = weights.effective_sample_size()
print(f"source ESS: {ess.source:.1f} / {len(weights.source)}")
print(f"target ESS: {ess.target:.1f} / {len(weights.target)}")
```

Rule of thumb: worry when `ESS < n/4` for either group (not a hard cutoff). A significant weighted result with healthy ESS is convincing; with `ESS ≈ 1` it may be driven by one or two points.

- Compare ESS to `n` for *each* group, not across groups.
- ESS is diagnostic, not a verdict — use with the weighted p-value.
- If ESS stays low even at `shrinkage=0.5`, groups barely overlap; skip weighting or collect more comparable data.

Sweep example:

```text
shrinkage=0.0   source ESS=   6.53  target ESS= 203.61
shrinkage=0.5   source ESS= 283.68  target ESS= 302.34
```

See [Diagnose weight concentration](../../examples/weighting/diagnose-weight-concentration.md) and [Importance weights](../../api/weighting.md#effective-sample-size).

Reference: Kish, L. (1965). *Survey Sampling*.
