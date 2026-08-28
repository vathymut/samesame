# Shift testing

Scores only — turn raw features into one numeric score per group first. Common scores: predicted risk, prediction error, or outlier score (`LogitGap`). `P(target|x)` is valid for `test_shift` but double-duty as harm score is discouraged when weighting — use it to build weights instead.

> Source: `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

!!! tip "Honest scores"

    If scores come from a fitted model, generate them out of sample with `cross_val_predict`, `oob_decision_function_`, or a held-out set. In-sample scores create false separation and invalidate the test.

## Which function?

| Function | Answers | When |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | you want any difference |
| `ss.test_harmful_shift` | Did target shift toward worse? | you can declare `worse` |

## Parameters

All functions take:

- `n_resamples` — permutation resamples. Default `9999`. Use `999` while exploring, `9999` final, `19999` for `p < 0.001` resolution.
- `rng` — `int` seed, `np.random.Generator`/`RandomState`, or `None`. Prefer `np.random.default_rng(12345)`.
- `weights` — `ImportanceWeights` from `ss.domain_weights`. Must match `len(source)`/`len(target)`.

`ss.test_harmful_shift` also requires `worse`:

--8<-- "snippets/worse-table.txt"

## Returns

- `ss.test_shift` → `ShiftResult` (`.statistic` = AUC, two-sided p)
- `ss.test_harmful_shift` → `HarmfulShiftResult` (`.statistic` = harm, one-sided `greater`, plus `.worse`)

Both carry `.null_distribution`. `.pvalue` ∈ `(0, 1]` with +1 smoothing; two-sided doubling capped at 1.

- `.pvalue` — evidence against the null. Small (≤ 0.05) = unlikely under no shift.
- `.statistic` — magnitude. AUC `0.5` = chance; harm has no fixed scale — compare observed vs null median (see [How the harm test works](../explanation/harmful-shift-statistic.md)).

Read `.pvalue` first, `.statistic` second.

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
