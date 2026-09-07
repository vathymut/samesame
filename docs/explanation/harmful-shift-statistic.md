# How the harm test works

## Why a separate test

A distribution can change without becoming worse. A generic shift test should
detect both changes, but monitoring often asks a narrower question: did the
target move toward outcomes that matter? The answer depends on the score being
monitored and on which direction is harmful.

Without loss of generality, assume that a higher score means more harmful or
riskier outcomes. The `worse` parameter handles scores whose harmful direction
is lower instead. Choose that direction from the meaning of the score, not from
the p-value.

## Tail-focused separation

For an oriented score `S` and threshold `t`, define

\[
\operatorname{FPR}(t) = P(S > t \mid \text{source}),
\qquad
\operatorname{TPR}(t) = P(S > t \mid \text{target}).
\]

The ordinary shift statistic is ROC AUC: it weights thresholds uniformly. The
harm statistic gives more weight to thresholds that the source rarely exceeds:

\[
\operatorname{AUC} = \int \operatorname{TPR}(t)\,d\operatorname{FPR}(t),
\qquad
T = \int \operatorname{TPR}(t)\,[1-\operatorname{FPR}(t)]^2\,
    d\operatorname{FPR}(t).
\]

Here, `T` is the weighted AUC: the test statistic used by
`test_harm`.
The statistic measures directional separation.

Since `1 - FPR(t)` is the source probability below the threshold, this
emphasizes target observations entering the source's unusual upper tail. AUC
treats all thresholds equally. The harm statistic puts more emphasis on target
observations beyond the source's usual range. This makes it more sensitive to
target observations entering regions rarely observed in source than to
movement among values commonly observed in benign source samples.

## How to choose the test

- Use `test_shift` when any distributional difference matters.
- Use `test_harm` when you can declare the harmful direction before
  looking at the result.

Both tests compare the same source and target scores by permutation. The
harmful-shift test changes the threshold weighting and uses a one-sided
alternative.

See [Shift testing](../api/testing.md) for the function signatures and
[Get started](../examples/tutorials/get-started.md) for a complete score and
test workflow.

## Reference

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts* (D-SOS).
Proceedings of UAI, PMLR 180:959-968.
[PMLR](https://proceedings.mlr.press/v180/kamulete22a.html) and
[arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
