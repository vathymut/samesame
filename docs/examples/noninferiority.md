# Noninferiority Test

Noninferiority tests (NITs) ask whether a new sample (or treatment)
is *not meaningfully worse* than a reference. `samesame` implements D-SOS
(Dataset Shift with Outlier Scores), a robust, nonparametric noninferiority
test that operates on outlier scores and does not require a pre-specified
margin.

## When to use this

- **Use CTSTs** when the question is, "are the
 two distributions different?" — i.e., you want to detect any distributional
 difference.
- **Use NITs** when the question is, "is the new
 sample *not substantially worse* than the reference?" — i.e., you care about
 adverse shifts rather than any difference.

The two approaches address different scientific questions; they are
complementary rather than interchangeable.

## D-SOS at a glance

[D-SOS](https://proceedings.mlr.press/v180/kamulete22a.html)
transforms the problem of noninferiority testing (NITs) into a CTST by treating
outlier scores as the classifier's predicted values, using a weighted AUC metric,
and testing a one-sided alternative. The key advantages are:

- Nonparametric: no normality assumption required.
- No pre-specified margin needed: the method is robust to how "meaningful"
 differences are defined in practice.
- Works with any sensible outlier scoring method (isolation forest, deep
 models, domain-specific scores, etc.).

The test focuses on whether the test sample contains *disproportionately
more* high outlier scores than the reference, which aligns with the goal of
detecting adverse shifts.

## Prologue

The following clinical-trial style example illustrates the difference between
classic noninferiority approaches and D-SOS. The motivating study (from SAS's
case study) compares a new, cheaper drug "Bowl" to the standard "Armanaleg."

The [original study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf) uses parametric assumptions and a pre-specified margin. For
D-SOS we only need outlier/discomfort scores that reflect worse outcomes as
higher values.

### Data

We convert reported relief scores so that higher numbers indicate *worse*
outcomes, producing outlier/discomfort scores suitable for D-SOS.

```python
import numpy as np

datalines = (
  "9 14 13 8 10 5 11 9 12 10 9 11 8 11 "
  "4 8 11 16 12 10 9 10 13 12 11 13 9 4 "
  "7 14 8 4 10 11 7 7 13 8 8 13 10 9 "
  "12 9 11 10 12 7 8 5 10 7 13 12 13 11 "
  "7 12 10 11 10 8 6 9 11 8 5 11 10 8"
).split()
relief = [float(s) for s in datalines]
discomfort = [max(relief) - s for s in relief]
armanaleg = np.array(discomfort[:28])
bowl = np.array(discomfort[28:])
```

## Analysis

Below we run D-SOS using `DSOS.from_samples`, treating `armanaleg` as the
reference (control) and `bowl` as the new treatment. We report the frequentist
p-value and an optional Bayesian conversion of the Bayes factor.

```python
from samesame.bayes import as_pvalue
from samesame.nit import DSOS

dsos = DSOS.from_samples(armanaleg, bowl)
frequentist = dsos.pvalue
bayesian = as_pvalue(dsos.bayes_factor)
print(f"Frequentist p-value: {frequentist:.4f}")
print(f"Bayesian p-value: {bayesian:.4f}")
```

Typical output (reproduced from the example):

```text
Frequentist p-value: 0.1215
Bayesian p-value: 0.1159
```

We fail to reject the null of *no adverse shift* — the new
treatment (Bowl) is not shown to be meaningfully worse than the reference
under the D-SOS criterion.

This is consistent with the original analysis from classical two-sided or
one-sided parametric noninferiority tests, which concludes:

> This suggests, as you’d hoped, that the efficacy of Bowl is not appreciably
> worse than that of Armanaleg

## Practical Recommendations

- Choose an outlier score that captures the phenomenon you care about (e.g.,
 clinical outcomes, high reconstruction error, extreme probability
 values, etc.).
- Consider reporting both distributional and noninferiority results when
 monitoring production systems: a distributional difference (e.g. CTST) does not
 always imply a practically meaningful or adverse change (e.g. DSOS).
