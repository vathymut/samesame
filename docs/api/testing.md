# Shift testing

Use this page when you already have a numeric signal for a source group and a target group.

## Choose the function

| Function | What it answers | Use it when |
|----------|------------------|-------------|
| `shift.detect_shift(...)` | Did anything change? | you want to detect any difference between source and target |
| `shift.detect_harm(...)` | Did the target group move in a worse direction? | you know what "worse" means for your signal |

Examples of useful signals include predicted risk, prediction error, model confidence, and domain
classifier probabilities.

## Common controls

All functions accept:

- `n_resamples` to control the number of permutation resamples
- `random_state` for reproducibility
- `weights` for weighted testing with `ImportanceWeights`

`shift.detect_harm(...)` also requires `direction`, a `Direction` enum member:

- `Direction.HIGHER_IS_WORSE` when larger scores mean harm
- `Direction.HIGHER_IS_BETTER` when larger scores mean quality

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult`
- `shift.detect_harm(...)` returns `HarmResult`

In each case, the fields most users look at first are:

- `.statistic`
- `.pvalue`

`ShiftResult` also includes `.statistic_name`.
`HarmResult` includes `.direction`.
All results include `.null_distribution` when you need the full permutation output.
All results offer `.significant(alpha=0.05)` to compare the p-value against a level.

## Posterior evidence for harmful shift

If you want posterior draws and a Bayes factor alongside the p-value, pass `bayesian=True` to
`shift.detect_harm(...)`.

```python
import samesame as ss

result = ss.detect_harm(
    source_scores,
    target_scores,
    direction=ss.Direction.HIGHER_IS_WORSE,
    bayesian=True,
)

print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {result.bayes_factor:.2f}")
```

`bayesian=True` returns `BayesianHarmResult`, which adds `.posterior` (the posterior draws) and
`.bayes_factor`. The optional `threshold` parameter sets the statistic value above which evidence
counts as harm; it defaults to `1 / 12` and requires `bayesian=True`.

## API

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

## Result types

::: samesame.shift.Direction

::: samesame.shift.TestResult

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult

::: samesame.shift.BayesianHarmResult
