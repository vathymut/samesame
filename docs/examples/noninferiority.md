# Noninferiority Test

Sometimes the pertinent question is, are we worse off? And when that is so,
statistical tests of equal distribution and of mean difference are not the
best tools for the job (see [here](https://vathymut.org/posts/2023-01-03-are-you-ok/)).

Noninferiority tests can help but these tests come with their own challenges. Typically, we need to define a pre-specified margin, the minimum meaningful difference needed to sound the alarm. This can be difficult, and controversial, to set in advance even for [domain experts](https://doi.org/10.1111/bcp.13280).

## D-SOS

Enter [D-SOS](https://proceedings.mlr.press/v180/kamulete22a.html), 
short for dataset shift with outlier scores. D-SOS is a robust
nonparametric noninferiority test that does not require a pre-specified
margin. It tests the null of no adverse shift based on outlier scores i.e.
it checks whether the new sample is not substantively worse than the old
sample, and not if the two are equal as tests of equal distributions do. This
two-sample comparison assumes that we have both a training set, the reference 
distribution of outlier scores, and a test set.

## Prologue

An example best illustrates how to use the method. The 
[case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf),
reproduced below, is from a clinical trial.

> You are a consulting statistician at a pharmaceutical company, charged with
> designing a study of your company’s new arthritis drug, SASGoBowlFor’Em
> (abbreviated as “Bowl”). Your boss realizes that Bowl is unlikely to demonstrate
> better efficacy than the gold standard, Armanaleg, but its lower cost will make
> it an attractive alternative for consumers as long as you can show that the
> efficacy is about the same.
>
> Your boss communicates the following study plans to you:
>> - The outcome to be measured is a “relief score,” which ranges from 0 to 20 and
>> is assumed to be approximately normally distributed.
>> - Subjects are to be allocated to Armanaleg and Bowl at a ratio of 2 to 3,
>> respectively.
>> - The relief score is to be assessed after four weeks on the treatment.
>> - Bowl is expected to be slightly less effective than Armanaleg, with a mean
>> relief score of 9.5 compared to 10 for Armanaleg.
>> - The minimally acceptable decrease in relief score is considered to be 2 units,
>> corresponding to a 20% decrease, assuming a mean relief score of 10 for Armanaleg.
>> - The standard deviation of the relief score is expected to be approximately
>> 2.25 for each treatment. Common standard deviation will be assumed in the data
>> analysis.
>> - The sample size should be sufficient to produce an 85% chance of a significant
>> result—that is, a power of 0.85—at a 0.05 significance level.


While quite a bit of this context is helpful and needed to run Schuirmann’s
classic method of two one-sided tests, this is not required for D-SOS. The
latter assumes no parametric form for the data (normality), and does not require
a pre-specified margin (2 units decrease in relief score).

## Data

D-SOS works with outlier scores so we turn these "relief scores" into
"discomfort scores" so that the higher the score, the worse the outcome.

```python
import numpy as np

datalines = "9 14 13 8 10 5 11 9 12 10 9 11 8 11 \
4 8 11 16 12 10 9 10 13 12 11 13 9 4 \
7 14 8 4 10 11 7 7 13 8 8 13 10 9 \
12 9 11 10 12 7 8 5 10 7 13 12 13 11 \
7 12 10 11 10 8 6 9 11 8 5 11 10 8".split()
relief = [float(s) for s in datalines]
discomfort = [max(relief) - s for s in relief]
armanaleg = np.array(discomfort[:28])
bowl = np.array(discomfort[28:])
```

## Analysis

To run the test, we specify `armanaleg` as the control (reference/first sample)
and `bowl` as the new treatment (second sample).

```python
from samesame.bayes import as_pvalue
from samesame.nit import DSOS
# alternatively: from samesame.nit import WeightedAUC as DSOS

dsos = DSOS.from_samples(armanaleg, bowl)
frequentist = dsos.pvalue
bayesian = as_pvalue(dsos.bayes_factor)
```
... And the results? Drumroll, please. We fail to reject the null of no
adverse shift. That is, we are not worse off with the new treatment.

```python
print(f"Frequentist p-value: {frequentist:.4f}")
print(f"Bayesian p-value: {bayesian:.4f}")
```

```
Frequentist p-value: 0.1215
Bayesian p-value: 0.1159
```

This is consistent with the original analysis which rejects the null 
(a different null than the D-SOS null!) with a p-value of $p = 0.0192$ and
concludes:

> This suggests, as you’d hoped, that the efficacy of Bowl is not appreciably
> worse than that of Armanaleg

## Epilogue

... did you catch it? Under the hood, the D-SOS test *is* a classifier
two-sample test (CTST). It uses outlier scores as predicted values, the
weighted AUC as the performance metric and tests against a one-sided
alternative (p-value) instead of a two-sided one.
