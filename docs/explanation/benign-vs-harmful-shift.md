# Benign vs harmful shift

Tests of equal distributions ask whether source and target are identical.
Monitoring usually needs a narrower question: is target substantively
worse than source? A shift can be detectable without being harmful, and a
harmful change can hide where a generic test is not looking.

## Why equality tests raise false alarms

Equal-distribution tests demand that the two samples match everywhere in
the input domain. At production scale that strictness backfires: tiny,
benign differences reject the null, teams learn to ignore the alarm, and
retraining on a benign shift is costly without improving outcomes.
Polyzotis et al. (2019) put it directly: distribution tests at ML scale
are "too sensitive and also uninformative." The harm statistic discounts
safe regions instead of weighting every threshold equally.

## A test for no adverse shift

D-SOS, the test behind `test_harm`, holds that target is not substantively
worse than source, not that the two samples are equal (Kamulete, 2022).
The question is reminiscent of clinical noninferiority: good enough, not
necessarily better, with no margin and no normality assumption.

You define worse with one score — risk, error, confidence, outlier score —
and a `worse` direction declared before testing. Higher risk is
`worse="higher"`. Lower confidence is `worse="lower"`. Pick that
direction from the score's meaning, never from the p-values.

The null is no adverse shift. A small p-value is evidence that target
moved into the harmful tail. A large p-value is not a certificate that
target is fine.

## What that focus costs

`test_shift` screens for any separation. `test_harm` reweights thresholds
toward the source's rare tail. The ROC form is in
[How the harm test works](harmful-shift-statistic.md). That focus is the
point, and the trade-off. A tail test can miss a shift that is not in
the harmful tail, including one you still care about. Run the two tests
separately. Do not read a detectable shift as harm.

## How the two tests fit

- Use `test_shift` when any distributional difference matters.
- Use `test_harm` with a pre-declared `worse` when only movement toward
  the harmful tail warrants intervention.
- Both permute group labels with scores fixed. A small p-value is
  evidence against exchangeability, not a measure of business impact.

See [How the harm test works](harmful-shift-statistic.md) for the ROC
derivation, [Get started](../examples/tutorials/get-started.md) for the
two-test workflow, [Is the new drug good enough?](../examples/trials/check-drug-efficacy.md)
for the same question with no model, and
[Monitor a credit model](../examples/credit/monitor-credit.md) for risk,
confidence, and error side by side.

## References

Kamulete, V. M. (2022). *Test for non-negligible adverse shifts.*
Proceedings of UAI, PMLR 180:959-968.
Polyzotis, N. et al. (2019). *Data validation for machine learning.*
Proc. MLSys 1:334-347.
