# Shift testing

Scores only — turn raw features into one numeric score per group first. See [Glossary](../explanation/glossary.md#score).

> Source: `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

--8<-- "snippets/honest-scores.txt"

## Choose a function

| Function | What it answers | Use it when |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | you want any difference between source and target |
| `ss.test_harmful_shift` | Did target shift toward worse outcomes? | you can declare what "worse" means |

Useful harm signals: predicted risk, prediction error (Brier/log-loss), or outlier scores (e.g., `LogitGap` confidence). Domain probability `P(target | x)` can be the score for `test_shift`, but don't reuse it as the harm score when you also weight — use it to build weights instead.

## Common controls

All functions accept `rng`, `weights`, and `n_resamples`:

--8<-- "snippets/n-resamples.txt"

- `rng` — reproducibility. Accepts `int` seed, `np.random.Generator`/`RandomState`, or `None`. Prefer a `Generator` (`np.random.default_rng(12345)`) for reproducibility; `int` is a shorthand.
- `weights` — `ImportanceWeights` from `ss.domain_weights` or constructed directly. Must match `len(source)` and `len(target)`.

`ss.test_harmful_shift` also requires `worse` — harmful direction (string or `ss.Worse`, interchangeable; enum gives autocomplete):

--8<-- "snippets/worse-table.txt"

## What you get back

- `ss.test_shift` → `ShiftResult` (`.statistic` is ROC AUC, two-sided p-value)
- `ss.test_harmful_shift` → `HarmfulShiftResult` (`.statistic` is harm statistic, one-sided `greater` p-value, plus `.worse`)

Both include `.null_distribution` when you need the full permutation output. `.pvalue` is in `(0, 1]` with +1 smoothing (Phipson & Smyth); two-sided doubling is capped at `1`.

--8<-- "snippets/pvalue-vs-statistic.txt"

See [Why the harm statistic is not just AUC](../explanation/harmful-shift-statistic.md) for the harm scale.

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
