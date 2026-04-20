# Advanced controls

Use this page when you need additional controls such as sample weights, more resamples,
the null distribution, or Bayesian evidence.

## What you get back

- `advanced.test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `advanced.test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, `.null_distribution`, and optional `.bayes_factor` and `.posterior`
- Both detailed result objects provide `.summary()` if you want the simpler primary result

::: samesame.advanced