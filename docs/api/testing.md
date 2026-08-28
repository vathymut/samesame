# Shift testing

These tests compare distributions of scores, not raw feature tables. First
turn each observation into a scalar that represents the question you care
about: predicted risk, prediction error, confidence, or an outlier score.

The distinction between the functions is important. `test_shift` detects any
change in the score distribution. `test_harmful_shift` deliberately ignores
some changes and looks for movement toward a declared harmful tail. Use the
first as a broad screen and the second when you can defend the meaning of
"worse" for the score.

`P(target|x)` from a domain classifier can be a useful score for detecting any
shift, but do not reuse it as the harm score. It describes how target-like an
observation is, not whether its outcome is harmful. Use it to build weights
when you need a common-support comparison.

> Source: `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

## Which function?

| Function | Answers | When |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | direction unknown |
| `ss.test_harmful_shift` | Did target shift toward worse? | you can declare `worse` |

`worse` is the polarity — string or `ss.Worse` (interchangeable):

--8<-- "snippets/worse-table.txt"

## Reading results

Both return `.statistic`, `.pvalue`, and `.null_distribution`. The
null distribution is produced by permuting group labels while keeping scores
and weights fixed. In other words, it represents what values are plausible if
the source and target labels were interchangeable.

Read `.pvalue` as evidence against that null, not as the probability that the
null is true and not as an effect size. Two-sided `test_shift` doubles the
smaller tail (capped at 1); `test_harmful_shift` uses the one-sided `greater`
alternative. A +1 correction keeps permutation p-values above zero.

- `test_shift`: `.statistic` is AUC. `0.5` means the score does not separate
  the groups; values farther from `0.5` indicate stronger separation.
- `test_harmful_shift`: `.statistic` has no universal interpretation like AUC.
  Compare it with `result.null_distribution` and examine the score scale. The
  result also records `.worse`.

For honest p-values, scores from a fitted model must be out of sample
(`cross_val_predict`, `oob_decision_function_`, or a held-out set). In-sample
predictions can make the source and target look more separable simply because
the scoring model has memorised its inputs.

??? tip "Reproducibility"
    `rng` accepts `int`, `np.random.Generator`/`RandomState`, or `None`. Prefer `rng=np.random.default_rng(12345)`. `n_resamples` default `9999` (`999` while exploring, `19999` for `p < 0.001`).

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
