# Benign vs harmful shift

Tests of equal distributions ask whether source and target are identical.
Monitoring usually needs a narrower question: is target substantively
worse than source? A shift can be detectable without being harmful, and a
harmful change can hide where a generic test is not looking.

## Context: why equality false-alarms

Equal-distribution tests demand fidelity everywhere in the input domain.
At production data scales that strictness backfires: tiny, benign
differences reject the null, teams learn to ignore the alarm, and
retraining on benign shifts is costly without improving outcomes.
Polyzotis et al. (2019) describe this directly — distribution tests at ML
scale are "too sensitive and also uninformative," which motivates a measure
that discounts safe regions instead of weighting every threshold equally.

## What: a noninferiority question

`samesame` reframes the check as noninferiority, borrowed from clinical
trials: the null is not "source equals target" but "target is not
substantively worse than source." What counts as worse is defined by one
score you care about — risk, error, confidence, outlier score — plus the
`worse` direction you declare before testing. Higher risk is `worse="higher"`;
lower confidence is `worse="lower"`. Pick that direction from the score's
meaning, never from the p-values.

## Why it works, and what it gives up

`test_shift` uses ROC AUC: it weights all thresholds uniformly and screens
for any separation. `test_harm` uses the weighted AUC
`∫ TPR·(1−FPR)² dFPR`: thresholds the source rarely exceeds get more
weight, so target mass entering the source's unusual tail counts more than
movement among values the source already produces often. That focus is the
point — and the trade-off. A harmful-tail test can miss a benign-looking
shift that still matters to you, which is why the two tests are run
separately instead of conflated.

## How it fits

- Use `test_shift` when any distributional difference matters.
- Use `test_harm` with a pre-declared `worse` when only movement toward
  the harmful tail warrants intervention.
- Both permute group labels with scores fixed; a small p-value is evidence
  against exchangeability, not a measure of business impact.

See [How the harm test works](harmful-shift-statistic.md) for the ROC
derivation, [Get started](../examples/tutorials/get-started.md) for the
two-test workflow, [Is the new drug good enough?](../examples/trials/check-drug-efficacy.md)
for noninferiority without a model, and
[Monitor a credit model](../examples/credit/monitor-credit.md) for risk,
confidence, and error side by side.

## References

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts.*
Proceedings of UAI, PMLR 180:959-968.
Polyzotis, N. et al. (2019). *Data validation for machine learning.*
Proc. MLSys 1:334-347.
