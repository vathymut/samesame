# Is the new drug good enough?

Medicine calls this **noninferiority**: a cheaper, faster, or more tolerable treatment need not beat the standard, it must be *not meaningfully worse*. The harmful-shift test is a nonparametric noninferiority test with no margin and no normality assumption (Kamulete, 2022).

The example is a classic [SAS case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf): **Bowl** (cheaper) versus **Armanaleg** (standard). Relief is 4 to 16 (higher is better); mean 9.4 versus 10.1 (42 and 28 patients). Not better, but is it *meaningfully worse*? You have one score per patient and no model. Pre-register `worse`, as you would choose one side before unblinding.

If you are new to `samesame`, start with [Get started](../tutorials/get-started.md).

## The data

Seventy relief scores, no model, no features. `samesame` sees one score per observation. **Source** is the standard (Armanaleg); **target** is the challenger (Bowl).

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

Relief is 4 to 16 where higher is better. Declare that directly. `ss.Worse` orients the score for you (`worse="lower"` means `S = -scores` under the hood, `src/samesame/shift.py:214`).

## The verdict

```python
import samesame as ss

rng = np.random.default_rng(12345)
harm = ss.test_harmful_shift(source=armanaleg, target=bowl, worse="lower", rng=rng)  # or ss.Worse.LOWER
print(f"Harm statistic: {harm.statistic:.4f}")  # → 0.1813
print(f"p-value:        {harm.pvalue:.4f}")     # → 0.1319

rng = np.random.default_rng(12345)
shift = ss.test_shift(source=armanaleg, target=bowl, rng=rng)
print(f"AUC {shift.statistic:.4f} p={shift.pvalue:.4f}")  # → 0.4171, 0.2548
```

Together:

- `test_shift` (AUC 0.42, p=0.25): little evidence the arms differ. AUC below 0.5 reflects lower relief in Bowl (0.5829 if you flip to discomfort, `1 − 0.4171`).
- `test_harmful_shift` (p=0.13): little evidence Bowl is meaningfully worse.

Same conclusion as the original parametric analysis, with no margin, no normality assumption, and only 70 observations.

## What the statistic is asking

The harm statistic `∫ TPR·(1−FPR)² dFPR` weights the harmful tail: **does Bowl push patients into low relief that Armanaleg rarely produces?** Larger `(1−FPR)²` where Armanaleg is rarest. See [How the harm test works](../../explanation/harmful-shift-statistic.md).

## How to read a non-rejection

A p-value of 0.13 is not a certificate of equivalence. It says the observed difference is not unusual if there were no meaningful harm.

- **Absence of evidence ≠ evidence of absence.** With 28 vs 42 patients the test may lack power; a larger study or wider deployment window could sharpen the verdict.
- **Direction is part of the protocol.** Here `worse="lower"` because lower relief is worse. Choosing direction after seeing p-values turns a pre-specified test into a search.

## Why a drug trial belongs in a monitoring guide

Every deployed model is a challenger drug: the standard (Armanaleg) is `source`, the challenger (Bowl) is `target`, a relief score is one interpretable score per observation, and “not meaningfully worse” is `test_harmful_shift`. The challenger passes when it stays close to the standard; when it does not, `samesame` shows you where and how.

For the same test on a model score, see [Monitor a credit model](../credit/monitor-credit.md). To reweight for common support, see [Weight for common support](../../how-to/weight-for-common-support.md) and [Core concepts](../../explanation/core-concepts.md).

For your own question about "not meaningfully worse," swap in your score and declare `worse` before you look.
