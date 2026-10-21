# How the harm test works

## Why a separate test

A distribution can change without becoming worse. See
[Benign vs harmful shift](benign-vs-harmful-shift.md) for false alarms, the
null of no adverse shift, and which test to use.

Take a higher score as more harmful. If lower scores are worse, pass
`worse="lower"`. The test flips the sign, then uses the same statistic.
Choose that direction from the meaning of the score, not from the p-value.

## Tail-focused separation

For an oriented score `S` and threshold `t`, define

\[
\operatorname{FPR}(t) = P(S > t \mid \text{source}),
\qquad
\operatorname{TPR}(t) = P(S > t \mid \text{target}).
\]

The ordinary shift statistic is ROC AUC. It weights thresholds uniformly.
The harm statistic gives more weight to thresholds the source rarely
exceeds:

\[
\operatorname{AUC} = \int \operatorname{TPR}(t)\,d\operatorname{FPR}(t),
\qquad
T = \int \operatorname{TPR}(t)\,[1-\operatorname{FPR}(t)]^2\,
    d\operatorname{FPR}(t).
\]

`T` is the weighted AUC that `test_harm` reports. Because `1 - FPR(t)` is
the source probability below the threshold, thresholds the source rarely
exceeds count more. Target observations that enter that rare tail move
`T` more than a shift among values the source already produces often.

Compare `T` with its permutation null, not with `0.5`, because the weights
change the scale.

## Same samples, different question

Both tests permute group labels on the same source and target scores.
`test_harm` changes the threshold weights and uses a one-sided
alternative.

Signatures are in [Shift testing](../api/testing.md). A complete workflow
is in [Get started](../examples/tutorials/get-started.md).

## Reference

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts* (D-SOS).
Proceedings of UAI, PMLR 180:959-968.
[PMLR](https://proceedings.mlr.press/v180/kamulete22a.html) and
[arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
