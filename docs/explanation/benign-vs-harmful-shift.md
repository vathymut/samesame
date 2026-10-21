# Benign vs harmful shift

A difference between the score distributions of the reference group (`source`)
and the group under review (`target`) can be detectable without being harmful.
Monitoring often needs a narrower question: did a score tied to the outcome
move in a direction marked harmful? One score can miss a change another reveals.

## Why equality tests can raise unhelpful alarms

Tests of equal score distributions ask whether source and target match; they
do not ask whether a difference matters. At production scale,
even a small, harmless difference can produce a small p-value. If every
such alarm prompts retraining, teams can spend effort without improving
outcomes. Polyzotis et al. (2019) put it directly: distribution tests at
ML scale are "too sensitive and also uninformative." D-SOS takes a narrower
view: it tests movement toward a tail you have marked harmful.

## A test for no adverse shift

D-SOS, which powers `test_harm`, tests the null of no adverse shift in the
direction you declared: target has not moved farther into the harmful tail.
It does not test whether source and target score distributions are identical
(Kamulete, 2022). The question is related to clinical noninferiority, but
D-SOS is not a clinical noninferiority test. It is nonparametric (it assumes
no particular score distribution) and uses no prespecified harm margin.

You define worse with one score — risk, error, confidence, outlier score —
and a `worse` direction declared before testing. Higher risk is
`worse="higher"`. Lower confidence is `worse="lower"`. Pick that
direction from the score's meaning, never from the p-values.

A small p-value is evidence against the no-adverse-shift null in the
direction you declared. A large p-value does not establish equivalence or
prove that target is fine.

## What that focus costs

`test_shift` uses ROC AUC to test whether a randomly chosen target score tends
to be higher than a randomly chosen source score more or less often than
chance. It is not an omnibus test: distributions can differ in shape while
their AUC remains at chance (`0.5`).
`test_harm` gives more weight to thresholds the source rarely exceeds. The
ROC form is in [How the harm test works](harmful-shift-statistic.md). That
focus is the point—and the trade-off. A tail test can miss a shift outside
the harmful tail, including one you still care about. Run the two tests
separately; a detectable shift is not automatically harmful.

## How the two tests fit

- Use `test_shift` as a two-sided screen for rank-based separation.
- Use `test_harm` with a pre-declared `worse` when only movement toward
  the harmful tail warrants intervention.
- Both hold scores fixed and permute group labels. A small p-value is evidence
  against label exchangeability: under the null, group labels are unrelated
  to the pooled scores. It is not a measure of business impact.

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
