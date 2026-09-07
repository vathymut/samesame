# Is the new drug good enough?

In clinical trials, this question is called **noninferiority**. A cheaper, faster, or easier-to-tolerate treatment does not have to beat the standard. It can win on cost or convenience, but it cannot leave patients meaningfully worse off. The harmful-shift test was introduced as D-SOS, a nonparametric noninferiority test with no margin and no normality assumption (Kamulete, 2022). See [Are you OK? Test for harmful (adverse) shift](https://vathymut.org/posts/2023-01-03-are-you-ok/) for the motivation behind the question.

Consider **Bowl**, a cheaper treatment, against **Armanaleg**, the established standard, in this [SAS case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf). Patients report relief on a 4-to-16 scale, with higher scores better. Bowl's mean relief is lower than Armanaleg's (9.4 versus 10.1), but a lower mean does not answer the practical question: did Bowl push more patients into relief levels the standard rarely produces? With one score per patient and no model, the harmful-shift test makes that question explicit. Set `worse` from the meaning of the score before looking at the results.

## The data

Load the 70 patient-level relief scores and split them into the reference (`source`) and current (`target`) samples. The first 28 observations are Armanaleg; the remaining 42 are Bowl.

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

Higher relief is better, so lower scores are harmful. Use `worse="lower"`.

## Run the two tests

```python
import samesame as ss

rng = np.random.default_rng(12345)
shift = ss.test_shift(source=armanaleg, target=bowl, rng=rng)
print(f"Shift p-value:  {shift.pvalue:.4f}")    # → 0.2548

rng = np.random.default_rng(12345)
harm = ss.test_harmful_shift(source=armanaleg, target=bowl, worse="lower", rng=rng)
print(f"Harm p-value:   {harm.pvalue:.4f}")     # → 0.1319
```

The two tests answer different questions:

- `test_shift` (p=0.25) tests for any difference between the source and target score distributions.
- `test_harmful_shift` (p=0.13) tests for movement toward the harmful, low-relief tail.

Neither p-value is small, so we do not reject the null of no shift or the null of no harmful shift.

That is consistent with the original parametric analysis ("not appreciably worse"), while making no normality assumption and using only 70 observations.

The harmful-shift test focuses on the harmful tail: it asks whether Bowl leaves more patients in low-relief territory that Armanaleg rarely visits. Differences where the standard already has many patients matter less. See [How the harm test works](../../explanation/harmful-shift-statistic.md) for the mathematical details and ROC intuition.

## What the non-rejection means

A p-value of 0.13 is not a certificate of equivalence or proof that Bowl is safe. It says the observed pattern would not be surprising under the null of no harmful shift.

- **Absence of evidence isn't evidence of absence.** With 28 versus 42 patients, the test may lack power; a larger study or wider deployment window could sharpen the verdict.
- **Direction is part of the protocol.** Here `worse="lower"` because lower relief is worse. Picking a direction after seeing p-values turns a pre-specified test into a search.
