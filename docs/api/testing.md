# Shift testing

Scores only — turn raw features into one numeric score per group first.

> Source: `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

--8<-- "snippets/honest-scores.txt"

## Choose a function

| Function | What it answers | Use it when |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | you want any difference |
| `ss.test_harmful_shift` | Did target shift toward worse outcomes? | you can declare `worse` |

Signals: predicted risk, prediction error, or outlier score (`LogitGap`). `P(target|x)` can be the score for `test_shift` but not for harm when weighting — use it to build weights instead.

## Common controls

All functions accept `rng`, `weights`, and `n_resamples`:

--8<-- "snippets/n-resamples.txt"

- `rng` — `int` seed, `np.random.Generator`/`RandomState`, or `None`. Prefer `np.random.default_rng(12345)`.
- `weights` — `ImportanceWeights` from `ss.domain_weights`. Must match `len(source)`/`len(target)`.

`ss.test_harmful_shift` also requires `worse` (string or `ss.Worse`):

--8<-- "snippets/worse-table.txt"

## What you get back

- `ss.test_shift` → `ShiftResult` (`.statistic` is AUC, two-sided p-value)
- `ss.test_harmful_shift` → `HarmfulShiftResult` (`.statistic` is harm, one-sided `greater`, plus `.worse`)

Both include `.null_distribution`. `.pvalue` is in `(0, 1]` with +1 smoothing; two-sided doubling capped at `1`.

--8<-- "snippets/pvalue-vs-statistic.txt"

See [How the harm test works](../explanation/harmful-shift-statistic.md) for the harm scale.

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
