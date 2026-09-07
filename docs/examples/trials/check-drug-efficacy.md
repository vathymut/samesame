# Is the new drug good enough?

Medicine calls this **noninferiority**: a cheaper, faster, or easier-to-tolerate treatment need not beat the standard. It has to be *not meaningfully worse*. The harmful-shift test was introduced as D-SOS, a nonparametric noninferiority test with no margin and no normality assumption (Kamulete, 2022).

The example is a classic [SAS case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf): **Bowl** (cheaper) versus **Armanaleg** (standard). Relief runs 4 to 16 (higher is better), with means of 9.4 versus 10.1 across 42 and 28 patients. Bowl doesn't look better, but *not better* isn't the question. The question is whether it's *meaningfully worse*. You have one score per patient and no model, so pre-register `worse` as you'd pick a side before unblinding.

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
print(f"p-value:        {harm.pvalue:.4f}")     # → 0.1319

rng = np.random.default_rng(12345)
shift = ss.test_shift(source=armanaleg, target=bowl, rng=rng)
print(f"Shift p-value:  {shift.pvalue:.4f}")    # → 0.2548
```

Together they tell one story:

- `test_shift` (p=0.25): little evidence the arms differ at all.
- `test_harmful_shift` (p=0.13): little evidence Bowl is meaningfully worse.

That matches the original parametric analysis ("not appreciably worse") with no margin, no normality assumption, and only 70 observations.

## What the statistic is asking

The harm statistic `∫ TPR·(1−FPR)² dFPR` leans into the harmful tail: **does Bowl leave more patients in low-relief territory that Armanaleg rarely visits?** Each threshold is a relief level, and `(1−FPR)²` weighs most where Armanaleg is rarest. If Bowl's worst cases bunch up there, the statistic grows; where the arms differ on ground the standard already covers, it stays modest, as here. See [How the harm test works](../../explanation/harmful-shift-statistic.md).

## How to read a non-rejection

A p-value of 0.13 is not a certificate of equivalence. It says the observed gap wouldn't be surprising if there were no meaningful harm.

- **Absence of evidence isn't evidence of absence.** With 28 versus 42 patients, the test may lack power; a larger study or wider deployment window could sharpen the verdict.
- **Direction is part of the protocol.** Here `worse="lower"` because lower relief is worse. Picking a direction after seeing p-values turns a pre-specified test into a search.

## Why a drug trial belongs in a monitoring guide

Every deployed model is a challenger drug: the standard arm is `source`, the deployed challenger is `target`, and "not meaningfully worse" is still `test_harmful_shift`. For the same test on a model score, see [Monitor a credit model](../credit/monitor-credit.md); for overlap, see [Weight for common support](../../how-to/weight-for-common-support.md).

For your own question, swap in your score and declare `worse` before you look.
