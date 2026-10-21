# Is the new drug good enough?

In a clinical trial, “good enough” often means noninferior. A cheaper, faster,
or better-tolerated treatment need not beat the standard, but it should not
leave patients meaningfully worse off.

D-SOS asks a related question: is there evidence of adverse shift? It is not
a clinical noninferiority test. It sets no prespecified harm margin and
assumes no particular score distribution (Kamulete, 2022). See [Are you OK?
Test for harmful (adverse) shift](https://vathymut.org/posts/2023-01-03-are-you-ok/)
for the motivation behind the question.

Consider **Bowl**, a cheaper treatment, against **Armanaleg**, the established
standard, in this [SAS case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf).
Patients report relief on a 4-to-16 scale; higher scores are better. Bowl's
mean is lower than Armanaleg's (9.4 versus 10.1), but that alone does not show
whether more patients had relief levels the standard rarely produces. With
one score per patient and no model, the harm test asks that tail-focused
question. Set `worse` from the score's meaning before looking at results.

## The data

Load the 70 patient-level relief scores and split them into the reference (`source`) and challenger (`target`) samples. The first 28 observations are Armanaleg; the remaining 42 are Bowl.

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
harm = ss.test_harm(source=armanaleg, target=bowl, worse="lower", rng=rng)
print(f"Harm p-value:   {harm.pvalue:.4f}")     # → 0.1319
```

The two tests answer different questions:

- `test_shift` (p=0.25) tests for any difference between the source and target score distributions.
- `test_harm` (p=0.13) tests for movement toward the harmful, low-relief tail.

Neither p-value is small, so we do not reject the null of no shift or the null of no harmful shift.

The result is consistent with the case study’s parametric conclusion (“not
appreciably worse”). This nonparametric test uses only 70 patients; non-rejection
does not establish noninferiority.

The harmful-shift test focuses on the harmful tail: it asks whether Bowl leaves more patients in low-relief territory that Armanaleg rarely visits. Differences where the standard already has many patients matter less. See [How the harm test works](../../explanation/harmful-shift-statistic.md) for the mathematical details and ROC intuition.

## What the non-rejection means

A p-value of 0.13 does not establish equivalence or prove that Bowl is safe. It says the data do not provide strong evidence against the null of no harmful shift.

- **No detected harm is not proof of no harm.** With 28 versus 42 patients, the test may have limited power; a larger study could give a clearer assessment of small harms.
- **Direction is part of the protocol.** Here `worse="lower"` because lower relief is worse. Picking a direction after seeing p-values turns a pre-specified test into a search.

Same two tests on a model score: [Get started](../tutorials/get-started.md).
