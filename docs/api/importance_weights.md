# Importance weights

## When to use importance weights

Use importance weights when you know that source and target have different feature
distributions — covariate shift — and you want the shift test to focus on the region
where both groups overlap rather than penalising samples that are simply foreign to the
other group. If you have no prior knowledge of covariate shift, omit `membership_prob`
and let the default `NoWeighting` behaviour apply.

## Choosing a mode

| Mode | What it does |
|------|--------------|
| `mode="source"` | Down-weights source samples foreign to target. Target samples keep unit weight. |
| `mode="target"` | Down-weights target samples foreign to source. Source samples keep unit weight. |
| `mode="both"` | Down-weights outliers in both groups; focuses the test on common support. |

`alpha_blend` controls numerical stability: `0.0` is the plain density ratio (IWERM); `1.0` is
uniform weights (no correction). The default `0.5` is a safe starting point.
For guidance on which mode fits your scenario, see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).

## Connecting weights to a shift test

You do not need to call `contextual_weights` directly in most cases. Pass `membership_prob`
and `mode` straight to `test_shift` or `test_adverse_shift` and the weights are computed
internally:

```python
import samesame

result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    membership_prob=pooled_membership_probs,
    mode="source",
)
```

If you need to inspect or pre-compute weights outside a test call, use
`samesame.weights.contextual_weights` directly. See [Weighting strategies](weighting.md) for
the full integration reference.

**For a worked end-to-end example**, see the tutorial
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the conceptual background — why density ratios can become extreme and how `alpha_blend`
tames them — see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).

---

::: samesame.weights