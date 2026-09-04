# Is the new drug good enough?

Medicine has a name for the question this guide answers: **noninferiority**. A
new treatment — cheaper, faster to make, easier to tolerate — does not have to
beat the standard. It has to be *not meaningfully worse*. The harmful-shift
test grew up answering exactly this question: it was introduced as D-SOS, a
nonparametric noninferiority test that needs no pre-specified margin and no
normality assumption (Kamulete, 2022).

The example comes from a classic case study, reprinted in the
[SAS proceedings](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf).
A new, less expensive drug — call it **Bowl** — challenges the established
standard, **Armanaleg**. Patients report relief on a 4-to-16 scale; higher is
better. Bowl does not look better: mean relief is 9.4 against Armanaleg's
10.1 (42 vs 28 patients). But *not better* is not the question. The question
is: is Bowl meaningfully worse?

Notice what that question is **not**. "Are the two drugs different?" is a
two-sided question, and a cheap answer. "Is the challenger not meaningfully
worse?" is a one-sided verdict with a protocol behind it. In a trial, the
direction is pre-registered. In monitoring, you pre-specify `worse` from the
meaning of the score — before you look at any p-value.

If you are new to `samesame`, start with [Get
started](../tutorials/get-started.md).

## The data

Seventy relief scores, two arms. `samesame` sees one score per observation and
nothing else — no model, no features. **Source** is the standard (Armanaleg,
the reference); **target** is the challenger (Bowl, the treatment under
evaluation).

```python
import numpy as np

datalines = (
  "9 14 13 8 10 5 11 9 12 10 9 11 8 11 "
  "4 8 11 16 12 10 9 10 13 12 11 13 9 4 "
  "7 14 8 4 10 11 7 7 13 8 8 13 10 9 "
  "12 9 11 10 12 7 8 5 10 7 13 12 13 11 "
  "7 12 10 11 10 8 6 9 11 8 5 11 10 8"
).split()
relief = np.array([float(s) for s in datalines])
armanaleg, bowl = relief[:28], relief[28:]  # source: standard, target: challenger
```

The harmful-shift test wants an orientation: larger = worse. Relief points the
other way, so flip it into a **discomfort score** — `max(relief) − relief` —
where a large value means the patient was left far from the best possible
relief.

```python
discomfort = relief.max() - relief           # flip: larger = worse
armanaleg_harm = discomfort[:28]
bowl_harm = discomfort[28:]
```

Flipping the score and declaring `worse="lower"` are interchangeable — the
test negates internally, so both give the same verdict. Declare the direction;
the arithmetic follows.

## The verdict

```python
import samesame as ss

rng = np.random.default_rng(12345)
harm = ss.test_harmful_shift(source=armanaleg_harm, target=bowl_harm, worse="higher", rng=rng)
print(f"Harm statistic: {harm.statistic:.4f}")  # → 0.1813
print(f"p-value:        {harm.pvalue:.4f}")     # → 0.1319

rng = np.random.default_rng(12345)
shift = ss.test_shift(source=armanaleg_harm, target=bowl_harm, rng=rng)
print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")  # → 0.5829, 0.2548
```

Read the two results together:

- `test_shift` (two-sided, AUC 0.58) finds no strong evidence the arms differ
  at all.
- `test_harmful_shift` (one-sided) finds no evidence that Bowl is meaningfully
  worse — p = 0.13, far from any conventional alarm level.

This matches the original parametric analysis of the same study, which
concluded:

> This suggests, as you'd hoped, that the efficacy of Bowl is not appreciably
> worse than that of Armanaleg.

The same conclusion, reached without a pre-specified margin, without
normality, and with 70 observations.

## What the statistic is asking

The harmful-shift statistic is a weighted AUC: `∫ TPR·(1−FPR)² dFPR`. In this
trial it reads as one question: **does Bowl leave more patients in relief
territory that Armanaleg rarely visits?**

Each threshold is a level of discomfort. `FPR` is how often the *standard*
exceeds it; the `(1−FPR)²` factor concentrates attention on thresholds the
standard almost never breaches. If Bowl's worst cases bunch beyond those
thresholds, the statistic grows. If the arms differ mainly where the standard
already has plenty of patients, the statistic stays modest — as it does here.
A plain AUC treats every threshold equally; the harm weighting decides that
*some differences matter more than others*. See [How the harm test
works](../../explanation/harmful-shift-statistic.md) for the formula and the
ROC-curve intuition.

## How to read a non-rejection

A p-value of 0.13 is **not** a certificate of equivalence. It says: with this
much data, the observed difference is not surprising under no meaningful
harm. Two cautions carry over from trials to monitoring:

- **Absence of evidence is not evidence of absence.** With 28 versus 42
  patients, the test may simply lack power. A larger study, or a larger
  deployment window, could sharpen the verdict.
- **The direction is part of the protocol.** `worse="higher"` here because
  larger discomfort is worse. Choosing the direction after seeing the p-values
  turns a verdict into a fishing expedition.

## Why a drug trial lives in a monitoring guide

Because every deployed model is a challenger drug. Table of the translation:

| Trial | Monitoring | `samesame` |
|---|---|---|
| Standard (Armanaleg) | Training data or past deployment | `source` |
| Challenger (Bowl) | Current deployment | `target` |
| Relief score | One interpretable score per observation | score |
| "Not meaningfully worse" | No harmful shift | `test_harmful_shift` |

Your deployment is Bowl. Your training data is Armanaleg. The drug passes when
it is *same enough* — and when it fails, `samesame` tells you so.

For the same test applied to a model score on a real credit benchmark, see
[Monitor a credit model](../credit/monitor-credit.md). For reweighting the
comparison to comparable applicants, see [Weight for common
support](../weighting/weight-for-common-support.md).
