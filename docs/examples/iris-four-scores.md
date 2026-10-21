# One split, four scores: the iris lesson

> The iris example from Kamulete (2022), *Test for non-negligible adverse
> shifts* (UAI). Plots, splits, and score definitions are in the R vignette
> [A 10-minute Crash Course](https://vathymut.github.io/dsos/articles/motivation.html).

One source-target split, read through four scores. Higher is worse
throughout: a larger score means further from the desired outcome, or less
like source.

## The setup

Train-test pairs come from two partitionings — random sampling, and an
in-distribution (densest-region) test set — each probed with four scores:
two-sample domain probability, density-based outlier score (isolation
forest), residual error (out-of-bag random-forest error), and prediction
uncertainty (standard error of the mean prediction). Only the first mirrors
a test of equal distributions; the rest capture other notions of worse.
Results are reported as `s = −log2(p)`: an `s` of `k` reads as `k` straight
coin flips against the null.

## Finding 1: random splits can false-alarm

Under random sampling, the domain classifier rejects strongly (`s ≈ 8`)
while residual, density, and confidence scores stay compatible with no
adverse shift. Relying on equal-distribution logic alone would have raised
an alarm on an essentially benign shift.

## Finding 2: dense points are not automatically safe

Under the in-distribution split, density-based detection stays quiet — the
points look familiar — yet residual and confidence tests reject. The
densest region overlaps the `versicolor`/`virginica` boundary where the
classifier discriminates poorly: inliers that are hard to predict. Trouble
can come from predictable-looking points, not just outliers.

## Takeaway for `samesame`

Equal-distribution tests cannot tell a benign shift from a harmful one.
The random-split false alarm and the quiet density test on hard inliers
show the same blind spot from opposite sides. No single score sees the whole
shift. Trust scores, residuals, density, prediction uncertainty: if you
can define the score, you can test it.

In `samesame`: pick the score from the harm you care about, declare
`worse` before testing, and run `test_shift` (any change) separately from
`test_harm` (adverse movement). Test for adverse shifts, not equality.
