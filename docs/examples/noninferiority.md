# Noninferiority Test (D-SOS)

Noninferiority asks: *is the new sample not meaningfully worse than the reference?* `samesame` implements D-SOS (Dataset Shift with Outlier Scores), a nonparametric test on outlier scores with a one-sided alternative.

## When to use

- **CTST (distribution difference)**: detect any distributional change
- **D-SOS (adverse shift check)**: detect whether the new sample is worse (more high outlier scores)

Use D-SOS when you care about *harmful* shifts rather than any difference.

## How D-SOS works

- Treat outlier scores as classifier outputs
- Use a weighted AUC with a one-sided alternative (more high scores = worse)
- No parametric assumptions or preset margin needed

## Example: clinical trial

The [original example](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf) compares two treatments for relief from leg discomfort. We repurpose the relief scores as outlier scores to test noninferiority of the Bowl treatment versus the Armanaleg (reference).
We flip relief into discomfort scores so that higher means worse (outlier) and test if the new treatment is not worse than the reference.

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

Run D-SOS treating `armanaleg` as control and `bowl` as the new treatment.

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

## Interpreting results

- Small p-value → evidence of adverse shift (new sample has more extreme outliers)
- Large p-value → insufficient evidence of being worse (noninferior)

## Practical tips

- Pick outlier scores aligned to “worse outcomes” (e.g., high error, high risk, discomfort)
- Pair with CTST: CTST says “different”, D-SOS says “meaningfully worse?”
- Use when labels are scarce: outlier scores can be model-based or proxy metrics
