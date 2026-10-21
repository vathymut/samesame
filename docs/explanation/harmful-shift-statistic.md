# How the harm test works

## Why a separate test

A score distribution can change without the score moving in a harmful
direction. See [Benign vs harmful shift](benign-vs-harmful-shift.md) for
the limits of shift tests, the null of no adverse shift, and which test to
use.

For this derivation, orient the score so larger values mean more harm. If
lower values are worse, pass `worse="lower"`; the test flips the sign
before applying the same statistic. Choose the direction from the score's
meaning, not from the p-value.

## Tail-focused separation

For an oriented score `S` and threshold `t`, define

\[
\operatorname{FPR}(t) = P(S > t \mid \text{source}),
\qquad
\operatorname{TPR}(t) = P(S > t \mid \text{target}).
\]

The ordinary shift statistic is the area under the receiver operating
characteristic (ROC) curve, or ROC AUC. It gives equal weight to increments
in false-positive rate. The harm statistic gives more weight to thresholds
the source rarely exceeds:

\[
\operatorname{AUC} = \int \operatorname{TPR}(t)\,d\operatorname{FPR}(t),
\qquad
T = \int \operatorname{TPR}(t)\,[1-\operatorname{FPR}(t)]^2\,
    d\operatorname{FPR}(t).
\]

`T` is the weighted AUC reported by `test_harm`. Since `1 - FPR(t)` is
the probability that a source score is at or below the threshold, the
weight is largest where few source observations exceed it. Target scores
that enter this rare tail affect `T` more than changes among values the
source often produces.

Compare `T` with its permutation null, not with `0.5`, because the weights
change the scale.

## Same samples, different question

Both tests permute group labels on the same source and target scores.
`test_harm` changes the threshold weights and uses a one-sided alternative.

Signatures are in [Shift testing](../api/testing.md). A complete workflow
is in [Get started](../examples/tutorials/get-started.md).

## Reference

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts* (D-SOS).
Proceedings of UAI, PMLR 180:959-968.
[PMLR](https://proceedings.mlr.press/v180/kamulete22a.html) and
[arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
