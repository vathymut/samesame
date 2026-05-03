# Importance weights

## When to use importance weights

Use importance weights when you know that source and target have different feature
distributions — covariate shift — and you want the shift test to focus on the region
where both groups overlap rather than penalising samples that are simply foreign to the
other group. If you have no prior knowledge of covariate shift, omit `weights`.

## Choosing a mode

| Mode | What it does |
|------|--------------|
| `mode="source"` | Down-weights source samples foreign to target. Target samples keep unit weight. |
| `mode="target"` | Down-weights target samples foreign to source. Source samples keep unit weight. |
| `mode="both"` | Down-weights outliers in both groups; focuses the test on common support. |

`lambda_` controls numerical stability: `0.0` is the plain density ratio (IWERM); `1.0` is
uniform weights (no correction). The default `0.5` is a safe starting point.
For guidance on which mode fits your scenario, see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).

## Connecting weights to a shift test

Call `contextual_weights` to build per-sample weights, then pass the result as `weights=`
to `test_shift` or `test_adverse_shift`:

```python
import samesame
from samesame.weights import contextual_weights

weights = contextual_weights(
    source_prob=source_membership_probs,
    target_prob=target_membership_probs,
    mode="source",
)
result = samesame.test_shift(
    source=source_scores,
    target=target_scores,
    weights=weights,
)
```

`source_prob` and `target_prob` are the membership probabilities for source and target samples
separately. The prior ratio is inferred automatically from their lengths.
See [Weighting strategies](weighting.md) for a quick-reference comparison of all three
approaches.

**For a worked end-to-end example**, see the tutorial
[Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).
For the conceptual background — why density ratios can become extreme and how `lambda_`
tames them — see
[Why importance weights stabilise shift detection](../explanation/importance-weights-rationale.md).

---

::: samesame.weights