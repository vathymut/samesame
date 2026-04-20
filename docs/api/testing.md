# Testing functions

Use this page for the two primary user-facing tests: `test_shift(...)` and `test_adverse_shift(...)`.
Start here if you are new to the package or want the simplest API surface.

## What you get back

- `test_shift(...)` returns `ShiftResult` with `.statistic`, `.pvalue`, and `.statistic_name`
- `test_adverse_shift(...)` returns `AdverseShiftResult` with `.statistic`, `.pvalue`, and `.direction`

If you need the null distribution, additional controls, or Bayesian output, see the advanced page.

::: samesame