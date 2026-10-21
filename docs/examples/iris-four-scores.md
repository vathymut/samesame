# Four scores, two splits: the iris lesson

> The iris example from Kamulete (2022), *Test for non-negligible adverse
> shifts* (UAI). Plots, splits, and score definitions are in the R vignette
> [A 10-minute Crash Course](https://vathymut.github.io/dsos/articles/motivation.html).

Two source-target splits, read through four scores. All four scores are
oriented so higher is worse; for domain probability, that means “more
target-like,” not harmful on its own.

## The setup

The iris data are split two ways: by random sampling and by selecting an
in-distribution test set from the densest region. Each split is assessed with
four scores:

- domain-classifier probability, for source-target separation;
- density-based outlier score from an isolation forest;
- residual error from out-of-bag random-forest predictions; and
- prediction uncertainty, measured by the standard error of the mean prediction.

The domain score measures group separation. The other three represent
potential harm. Results use `s = −log2(p)`. An `s` of `k` means `p = 2⁻ᵏ`,
like `k` consecutive heads in a fair coin toss.

## Finding 1: a shift flag can be benign

Under random sampling, the test on the domain-classifier score rejects
strongly (`s ≈ 8`), while tests of residual, density, and
prediction-uncertainty scores are compatible with no adverse shift. A shift
test flags the difference but does not say whether it is harmful.

## Finding 2: dense points are not automatically safe

Under the in-distribution split, density-based detection stays quiet—the
points look familiar—yet residual-error and prediction-uncertainty tests
reject the null of no adverse shift. The densest region overlaps the
`versicolor`/`virginica` boundary, where the classifier struggles to
distinguish the classes.
Dense points are not automatically safe; monitoring only for outliers can
miss them.

## Takeaway for `samesame`

`test_shift` detects rank-based score separation, not whether it is harmful.
The random-split flag and the quiet density test on hard inliers show why no
single score sees the whole shift. Choose a score tied to the harm you care
about, declare `worse` before testing, and run `test_shift` for rank separation
separately from `test_harm` for movement toward the harmful tail.
