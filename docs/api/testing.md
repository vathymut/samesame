# Shift detection

Use this page for the two primary user-facing calls:
`shift.detect_shift(...)` and `shift.detect_harm(...)`.
Start here if you are new to the package or want the simplest API surface.

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `shift.detect_harm(...)` returns `HarmResult` with `.statistic`, `.pvalue`, `.direction`, and `.null_distribution`

For Bayesian output or advanced controls, see the [advanced page](advanced.md).

::: samesame.shift