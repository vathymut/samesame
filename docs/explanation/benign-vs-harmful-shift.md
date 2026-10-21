# Benign vs harmful shift

A change in score distribution can be detectable without being harmful.
Monitoring often needs a narrower question: did target move in a harmful
direction on a score tied to the outcome you care about? One score can
miss a change that another reveals.

## Why equality tests can raise unhelpful alarms

Tests of equality ask whether the source and target score distributions
match; they do not ask whether a difference matters. At production scale,
even a small, harmless difference can produce a small p-value. If every
such alarm prompts retraining, teams can spend effort without improving
outcomes. Polyzotis et al. (2019) put it directly: distribution tests at
ML scale are "too sensitive and also uninformative." The harm statistic
gives less weight to thresholds the source often exceeds.

## A test for no adverse shift

D-SOS, the test behind `test_harm`, tests the null of no adverse shift:
target has not moved toward the harmful tail you declared. It does not test
whether source and target score distributions are identical (Kamulete,
2022). The question resembles clinical noninferiority—is target meaningfully
worse?—but D-SOS is not a margin-based clinical test. It is nonparametric
and assumes no particular score distribution.

You define worse with one score — risk, error, confidence, outlier score —
and a `worse` direction declared before testing. Higher risk is
`worse="higher"`. Lower confidence is `worse="lower"`. Pick that
direction from the score's meaning, never from the p-values.

The null is no adverse shift. A small p-value is evidence against that
null, in the direction you declared. A large p-value does not establish
equivalence or prove that target is fine.

## What that focus costs

`test_shift` screens for any separation between the score distributions.
`test_harm` gives more weight to thresholds the source rarely exceeds. The
ROC form is in [How the harm test works](harmful-shift-statistic.md). That
focus is the point—and the trade-off. A tail test can miss a shift outside
the harmful tail, including one you still care about. Run the two tests
separately; a detectable shift is not automatically harmful.

## How the two tests fit

- Use `test_shift` when any difference in score distributions matters.
- Use `test_harm` with a pre-declared `worse` when only movement toward
  the harmful tail warrants intervention.
- Both permute group labels with scores fixed. A small p-value is evidence
  against label exchangeability (under the null, source and target labels
  are interchangeable); it is not a measure of business impact.

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
