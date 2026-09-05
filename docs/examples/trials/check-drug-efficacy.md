# Is the new drug good enough?

Medicine has a name for the question this guide explores: **noninferiority**. A new treatment — cheaper, faster to make, or easier to tolerate — does not need to beat the established standard. It needs to be *not meaningfully worse*. The harmful-shift test was introduced for exactly this setting: a nonparametric noninferiority test that requires no pre-specified margin and no normality assumption (Kamulete, 2022).

The example comes from a classic case study reprinted in the [SAS proceedings](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf). A new, less expensive drug — call it **Bowl** — is compared with the established standard, **Armanaleg**. Patients report relief on a scale from 4 to 16, where higher is better. Bowl does not look better on average: mean relief is 9.4 for Bowl versus 10.1 for Armanaleg (42 and 28 patients, respectively). But *not better* is not the question here. The question is whether Bowl is meaningfully worse.

Notice what this question is not. “Are the two drugs different?” is a two-sided comparison that is easy to ask and easy to answer. “Is the challenger not meaningfully worse?” is a one-sided judgment that depends on a declared direction. In a trial that direction is pre-registered; in monitoring you pre-specify `worse` from the meaning of the score, before you look at any p-value.

If you are new to `samesame`, start with [Get started](../tutorials/get-started.md).

## The data

Seventy relief scores from two arms. `samesame` sees one score per observation and nothing else — no model, no features. **Source** is the standard (Armanaleg, the reference); **target** is the challenger (Bowl, the group under evaluation).

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

The harmful-shift test is oriented so that larger values mean worse outcomes. Relief runs the other way, so we flip it into a **discomfort score**, `max(relief) − relief`, where a large value means the patient remained far from the best possible relief.

```python
discomfort = relief.max() - relief           # flip: larger = worse
armanaleg_harm = discomfort[:28]
bowl_harm = discomfort[28:]
```

Flipping the score and declaring `worse="lower"` are equivalent: the test negates internally, so both choices lead to the same verdict. Declare the direction that matches the meaning of the score and let the arithmetic follow.

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

- `test_shift` (two-sided, AUC 0.58) finds little evidence that the arms differ at all.
- `test_harmful_shift` (one-sided) finds little evidence that Bowl is meaningfully worse, with p = 0.13 — well above conventional thresholds for concern.

This matches the original parametric analysis of the same study, which concluded:

> This suggests, as you'd hoped, that the efficacy of Bowl is not appreciably worse than that of Armanaleg.

The same conclusion follows without a pre-specified margin, without a normality assumption, and with only 70 observations.

## What the statistic is asking

The harmful-shift statistic is a weighted AUC, `∫ TPR·(1−FPR)² dFPR`, that focuses on the harmful tail. In this trial it asks a single question: **does Bowl push more patients into discomfort that Armanaleg rarely produces?**

For any discomfort threshold, `FPR` is the fraction of Armanaleg patients above it, so `(1−FPR)²` is largest where Armanaleg is rarest. The statistic grows when Bowl's worst outcomes cluster beyond those rare thresholds; it stays modest when the arms differ only where Armanaleg is already common, as they do here. A standard AUC weighs every threshold equally; the harm statistic does not. See [How the harm test works](../../explanation/harmful-shift-statistic.md) for the formula and ROC intuition.

## How to read a non-rejection

A p-value of 0.13 is not a certificate of equivalence. It says that, with this much data, the observed difference is not unusual if there were no meaningful harm. Two cautions from trials carry over to monitoring:

- **Absence of evidence is not evidence of absence.** With 28 versus 42 patients, the test may lack power. A larger study — or a wider deployment window — could sharpen the verdict.
- **The direction is part of the protocol.** Here `worse="higher"` because larger discomfort is worse. Choosing the direction after seeing the p-values turns a pre-specified comparison into a search for significance.

## Why a drug trial belongs in a monitoring guide

Every deployed model is a challenger drug in this analogy. The correspondence is:

| Trial | Monitoring | `samesame` |
|---|---|---|
| Standard (Armanaleg) | Training data or past deployment | `source` |
| Challenger (Bowl) | Current deployment | `target` |
| Relief score | One interpretable score per observation | score |
| "Not meaningfully worse" | No harmful shift | `test_harmful_shift` |

The challenger passes when it remains close enough to the standard; when it does not, `samesame` helps you see where and how.

For the same test applied to a model score on a real credit benchmark, see [Monitor a credit model](../credit/monitor-credit.md). To compare the deployment with reweighting for common support, see [Weight for common support](../weighting/weight-for-common-support.md).
