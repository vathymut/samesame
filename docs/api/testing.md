# Shift testing

Use this page when you already have a numeric signal for a source group and a target group.

## Choose the function

| Function | What it answers | Use it when |
|----------|------------------|-------------|
| `shift.detect_shift(...)` | Did anything change? | you want to detect any difference between source and target |
| `shift.detect_harm(...)` | Did the target distribution shift toward worse outcomes? | you know what "worse" means for your signal |

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

`HarmResult` also includes `.direction`.
All results include `.null_distribution` when you need the full permutation output.
All results offer `.significant(alpha=0.05)` to compare the p-value against a level.

## API

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

## Result types

::: samesame.shift.Direction

::: samesame.shift.TestResult

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult
