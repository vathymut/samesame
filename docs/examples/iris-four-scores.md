# One split, four scores: the iris lesson

> This is a summary of the motivating example from Kamulete (2022), *Test
> for non-negligible adverse shifts* (UAI). Full plots, splits, and score
> definitions are in the R vignette
> [A 10-minute Crash Course](https://vathymut.github.io/dsos/articles/motivation.html);
> what follows is only the lesson.

The example runs one idea end to end on the iris data: the
same source-target split looks different through different scores. Higher
is worse throughout — a larger score means further from the desired outcome
or less conforming to source.

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

The vignette's conclusion is twofold. First, tests of equal distributions
are blind by definition to whether a shift is benign or harmful — the
random-split false alarm and the quiet density test on hard inliers both
show it. Second, a family of tests over complementary notions of
outlyingness gives a holistic view: no single score sees the whole shift,
and the framework extends to any score you can define, from trust scores to
prediction uncertainty.

In `samesame` terms: pick the score from the harm you care about, declare
`worse` in advance, and run `test_shift` (any change) separately from
`test_harm` (adverse movement). Test for adverse shifts, not equality.
