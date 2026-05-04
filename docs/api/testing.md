# Testing functions

Use this page for the two primary user-facing tests: `test_shift(...)` and `test_adverse_shift(...)`.
Start here if you are new to the package or want the simplest API surface.

## What you get back

- `test_shift(...)` returns `ShiftDetails` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `test_adverse_shift(...)` returns `AdverseShiftDetails` with `.statistic`, `.pvalue`, `.direction`, and `.null_distribution`

For Bayesian output or advanced controls, see the [advanced page](advanced.md).

::: samesame