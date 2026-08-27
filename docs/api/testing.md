# Shift testing

Use this page when you already have a numeric signal for a source group and a target group.

## Choose the function

| Function | What it answers | Use it when |
|----------|------------------|-------------|
| `shift.test_shift(...)` | Did anything change? | you want to detect any difference between source and target |
| `shift.test_harmful_shift(...)` | Did the target distribution shift toward worse outcomes? | you know what "worse" means for your signal |

Examples of useful signals include predicted risk, prediction error, model confidence, and domain
classifier probabilities.

## Common controls

All functions accept:

- `n_resamples` to control the number of permutation resamples
- `rng` for reproducibility; pass a NumPy random generator or leave it as `None`
- `weights` for weighted testing with `ImportanceWeights`

`shift.test_harmful_shift(...)` also requires `worse`:

- `worse="higher"` when larger scores mean harm
- `worse="lower"` when smaller scores mean harm

## What you get back

- `shift.test_shift(...)` returns `ShiftResult`
- `shift.test_harmful_shift(...)` returns `HarmfulShiftResult`

In each case, the fields most users look at first are:

- `.statistic`
- `.pvalue`

`HarmfulShiftResult` also includes `.worse`.
All results include `.null_distribution` when you need the full permutation output.

When scores come from a fitted model, generate them out of sample with cross-validation,
out-of-bag predictions, or a held-out evaluation set. In-sample predictions can create artificial
separation and invalidate the test interpretation. `samesame` receives scores only and cannot check
how they were generated.

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult
