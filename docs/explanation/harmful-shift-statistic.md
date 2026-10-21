# How the harm test works

## Why a separate test

An observed score shift may be harmless. This page explains `test_harm`, which
compares a reference group (`source`) with a group under review (`target`). See
[Benign vs harmful shift](benign-vs-harmful-shift.md) for the null of no adverse
shift and guidance on which test to use.

For this derivation, recode the score so larger values mean more harm. If
lower values are worse, pass `worse="lower"`; the test flips the sign before
applying the same statistic. Choose the direction from the score's meaning,
not from the p-value.

## Tail-focused separation

For the recoded score `S` and threshold `t`, define the false-positive rate
(`FPR`, the share of source scores above `t`) and true-positive rate (`TPR`,
the share of target scores above `t`):

\[
\operatorname{FPR}(t) = P(S > t \mid \text{source}),
\qquad
\operatorname{TPR}(t) = P(S > t \mid \text{target}).
\]

The ordinary shift statistic is ROC AUC, the area under the receiver operating
characteristic curve. It gives equal weight to increases in `FPR`. The harm
statistic gives more weight to thresholds the source rarely exceeds:

\[
\operatorname{AUC} = \int \operatorname{TPR}(t)\,d\operatorname{FPR}(t),
\qquad
T = \int \operatorname{TPR}(t)\,[1-\operatorname{FPR}(t)]^2\,
    d\operatorname{FPR}(t).
\]

`T` is the weighted AUC reported by `test_harm`. Since `1 - FPR(t)` is
the share of source scores at or below the threshold, the weight is largest
where few source observations exceed it. Target scores that enter this rare
tail affect `T` more than changes among values the source often produces.

Compare `T` with its permutation null—the distribution from shuffling group
labels—not with `0.5`, because the weights change the scale.

## Same samples, different question

Both tests hold scores fixed and permute the source and target labels.
`test_harm` uses the tail-focused threshold weights above and a one-sided
alternative.

Signatures are in [Shift testing](../api/testing.md). A complete workflow
is in [Get started](../examples/tutorials/get-started.md).

## Reference

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts* (D-SOS).
Proceedings of UAI, PMLR 180:959-968.
[PMLR](https://proceedings.mlr.press/v180/kamulete22a.html) and
[arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
