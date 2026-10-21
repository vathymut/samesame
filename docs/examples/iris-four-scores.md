# Four scores, two splits: the iris lesson

> The iris example from Kamulete (2022), *Test for non-negligible adverse
> shifts* (UAI). Plots, splits, and score definitions are in the R vignette
> [A 10-minute Crash Course](https://vathymut.github.io/dsos/articles/motivation.html).

Two source-target splits, read through four scores. Higher is worse
throughout: a larger score means further from the desired outcome, or less
like source.

## The setup

Train-test pairs come from two partitionings—random sampling and an
in-distribution (densest-region) test set—each assessed with four scores:
domain-classifier probability, density-based outlier score (isolation
forest), residual error (out-of-bag random-forest error), and prediction
uncertainty (standard error of the mean prediction). The domain score
measures source-target separation; the other three capture different
notions of potential harm. Results use `s = −log2(p)`: an `s` of `k`
corresponds to `p = 2⁻ᵏ`, like `k` straight heads in a fair coin toss.

## Finding 1: a shift flag can be benign

Under random sampling, the domain classifier rejects strongly (`s ≈ 8`),
while residual, density, and prediction-uncertainty scores stay compatible
with no adverse shift. A generic shift test would flag the difference but
would not say whether it is harmful.

## Finding 2: dense points are not automatically safe

Under the in-distribution split, density-based detection stays quiet—the
points look familiar—yet residual-error and prediction-uncertainty tests
reject the null of no adverse shift. The densest region overlaps the
`versicolor`/`virginica` boundary, where the classifier discriminates
poorly: inliers that are hard to predict. Dense points are not automatically
safe; monitoring only for outliers can miss them.

## Takeaway for `samesame`

`test_shift` detects rank-based score separation, not whether it is harmful.
The random-split flag and the quiet density test on hard inliers show the
same blind spot from opposite sides. No single score sees the whole shift.
Trust scores, residuals, density, prediction uncertainty: if you can
define the score, you can test it.

In `samesame`, pick a score from the harm you care about, declare `worse`
before testing, and run `test_shift` (rank-based separation) separately from
`test_harm` (movement in the harmful direction).
