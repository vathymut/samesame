# Shift testing

Scores only — turn raw features into one score per group first (predicted risk, prediction error, or outlier score). `P(target|x)` is valid for `test_shift`; don't reuse it as the harm score — use it to build weights instead.

> Source: `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

## Which function?

| Function | Answers | When |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | direction unknown |
| `ss.test_harmful_shift` | Did target shift toward worse? | you can declare `worse` |

`worse` is the polarity — string or `ss.Worse` (interchangeable):

--8<-- "snippets/worse-table.txt"

## Reading results

Both return `.statistic`, `.pvalue`, `.null_distribution`. Read `.pvalue` first (≤ 0.05 is evidence against the null), `.statistic` second. Two-sided `test_shift` doubles the smaller tail (capped at 1); `test_harmful_shift` is one-sided `greater`. +1 smoothing, so `.pvalue` ∈ `(0, 1]` — see `src/samesame/_permutation.py:67`.

- `test_shift`: `.statistic` is AUC (`0.5` = chance).
- `test_harmful_shift`: `.statistic` has no fixed scale — compare observed vs `result.null_distribution` median (see [How the harm test works](../explanation/harmful-shift-statistic.md)). Also carries `.worse`.

Labels are permuted; scores and weights stay fixed. For honest p-values, scores from a fitted model must be out of sample (`cross_val_predict`, `oob_decision_function_`, or held-out).

??? tip "Reproducibility"
    `rng` accepts `int`, `np.random.Generator`/`RandomState`, or `None`. Prefer `rng=np.random.default_rng(12345)`. `n_resamples` default `9999` (`999` while exploring, `19999` for `p < 0.001`).

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
