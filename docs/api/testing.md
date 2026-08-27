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
- `batch` to control how many permutations are processed at once. It changes
  peak memory use and runtime, not the number of resamples.
- `rng` for reproducibility
- `weights` for weighted testing with `ImportanceWeights`

`shift.detect_harm(...)` also requires `worse`:

- `worse="higher"` when larger scores mean harm
- `worse="lower"` when smaller scores mean harm

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult`
- `shift.detect_harm(...)` returns `HarmResult`

In each case, the fields most users look at first are:

- `.statistic`
- `.pvalue`

`HarmResult` also includes `.worse`.
All results include `.null_distribution` when you need the full permutation output.
All results offer `.significant(alpha=0.05)` to compare the p-value against a level.

When scores come from a fitted model, generate them out of sample with cross-validation,
out-of-bag predictions, or a held-out evaluation set. In-sample predictions can create artificial
separation and invalidate the test interpretation. `samesame` receives scores only and cannot check
how they were generated.

For large inputs, set `batch` to a positive integer to limit peak memory. Leaving it as `None`
uses SciPy's default and processes all resamples in one batch.

## API

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

## Result types

::: samesame.shift.TestResult

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult
