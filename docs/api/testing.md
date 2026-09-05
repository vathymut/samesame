# Shift testing

Reduce each observation to one interpretable score, such as predicted risk,
prediction error, confidence, or an outlier score. Compare its distribution
between **source** (reference) and **target** (current deployment) rather than
a raw feature table. Choose a score that represents the question you care
about and compute it for every observation.

`test_shift` is a broad, two-sided screen (AUC `∫ TPR dFPR`, `0.5` is
chance). `test_harmful_shift` is a focused, one-sided tail test (weighted AUC
`∫ TPR·(1−FPR)² dFPR`) that emphasizes thresholds the source rarely
exceeds. Use the first when any change matters; the second when you can
pre-specify `worse`.

`P(target|x)` from a domain classifier can be a useful score for detecting any
shift, but do not reuse it as the harm score. It describes how target-like an
observation is, not whether its outcome is harmful. Use it to build weights
when you need a common-support comparison.

??? details "Source files"
    `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

## Which function?

| Function | Answers | When |
|----------|---------|------|
| `ss.test_shift` | Do source and target differ? | Any score distributional change matters |
| `ss.test_harmful_shift` | Did target move toward the specified harmful tail? | You can declare `worse` in advance |

--8<-- "snippets/worse-declaration.txt"

--8<-- "snippets/worse-table.txt"

## Reading results

Both return `.statistic`, `.pvalue`, and `.null_distribution`. The null
distribution is produced by permuting group labels while keeping scores and
weights fixed. It represents what values are plausible if the source and
target labels were interchangeable.

Interpret `.pvalue` as evidence against label exchangeability, not as the
probability that the null is true and not as an effect size. Two-sided
`test_shift` doubles the smaller tail, capped at 1; `test_harmful_shift` uses
the one-sided `greater` alternative. A `+1` correction keeps permutation
p-values above zero (Phipson & Smyth, 2010).

- `test_shift`: `.statistic` is AUC. An AUC of `0.5` means the score does not
  separate the groups; values farther from `0.5` indicate stronger separation.
- `test_harmful_shift`: `.statistic` is weighted AUC; read it against
  `result.null_distribution` and the score's own scale (see [How the harm
  test works](../explanation/harmful-shift-statistic.md)). The result also
  records `.worse`.

Honest p-values require out-of-sample scores. `samesame` only sees scores, not how they were made.

--8<-- "snippets/honest-scores.txt"

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` to make the permutation-based p-values reproducible. The default is `n_resamples=9999`; `999` is useful while exploring, while `19999` gives better resolution for p-values below `0.001`.

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
