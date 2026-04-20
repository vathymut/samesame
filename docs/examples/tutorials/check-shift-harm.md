# Tutorial: Check whether a shift is harmful

This tutorial is a guided first run of `test_adverse_shift(...)`.
You will start from a score where larger values mean better outcomes, set direction correctly,
and interpret whether the candidate sample is meaningfully worse.

**By the end, you will be able to:**

- The difference between detecting any shift and detecting a *harmful* shift
- How to use `test_adverse_shift(...)` and tell it which direction means worse
- How to read the p-value and decide whether the shift matters

In practice, this test complements `test_shift(...)`: first detect change, then assess harm.

## Why difference alone is not enough

Imagine you deploy a machine learning model in January and again in February.
You run `test_shift(...)` and get a small p-value — the February data is statistically
different from January. Should you be worried?

Not necessarily. Any two real-world datasets will differ slightly due to random variation.
Shift testing can detect even small, practically unimportant differences.

The more decision-relevant question is: **"Has February's data shifted in a way that could
harm the model or the downstream process?"** That is the question `test_adverse_shift(...)` answers.

## Shift vs. Adverse Shift at a glance

| Test  | Question it answers                          | When to use it                              |
|-------|----------------------------------------------|---------------------------------------------|
| `test_shift` | Are the two distributions different?         | Any time you want to detect *any* change |
| `test_adverse_shift` | Is the new data *worse* than the reference?  | When you care about *harmful* shifts only |

Use both together: `test_shift` to detect change, `test_adverse_shift` to judge severity.

Like the shift test, this procedure works on one score per sample rather than the full table.

## What kind of input does this test need?

`test_adverse_shift` needs one score per sample that encodes how adverse that sample is.
Larger values must correspond to "worse". Examples:

- A model's predicted probability of failure or default
- An anomaly level from a detector
- A patient's discomfort or risk measure

The test asks whether the new dataset contains a disproportionate share of the large values.

## How the test works

1. Pool both datasets and mark which samples belong to each group.
2. Compute how concentrated the largest values are in the new dataset.
3. Assess statistical significance via a **permutation test**: how often does random reassignment of group labels produce a concentration as extreme as the one observed? A small p-value means the observed concentration is unlikely under the null.

No distributional assumption is required, and no threshold needs to be specified in advance.

## Example: comparing two treatments

This example compares two treatments for relief from leg discomfort: *Armanaleg* (the established
reference) and *Bowl* (the new treatment). The scores measure discomfort — higher means more
discomfort, which is worse.

We want to know: **is the Bowl treatment meaningfully worse than Armanaleg?**

### Step 1 — Load the data

Relief scores are converted to discomfort scores internally, via `direction="higher-is-better"`.

```python
import numpy as np

relief = np.array([
     9, 14, 13,  8, 10,  5, 11,  9, 12, 10,  9, 11,  8, 11,
     4,  8, 11, 16, 12, 10,  9, 10, 13, 12, 11, 13,  9,  4,
     7, 14,  8,  4, 10, 11,  7,  7, 13,  8,  8, 13, 10,  9,
    12,  9, 11, 10, 12,  7,  8,  5, 10,  7, 13, 12, 13, 11,
     7, 12, 10, 11, 10,  8,  6,  9, 11,  8,  5, 11, 10,  8,
])
armanaleg  = relief[:28]  # reference treatment
bowl       = relief[28:]  # new treatment
```

### Step 2 — Run the adverse-shift test

Pass the reference sample first, then the new sample. `test_adverse_shift(...)` tests whether `bowl`
contains disproportionately more high-discomfort (worse) cases than `armanaleg`:

```python
from samesame import test_adverse_shift

harm = test_adverse_shift(
  reference=armanaleg,
  candidate=bowl,
  direction="higher-is-better",
)

print(f"Adverse-shift p-value: {harm.pvalue:.4f}")
```

**Output:**

```text
Adverse-shift p-value: 0.1215
```

## Reading the results

| p-value         | What it means                                                         |
|-----------------|-----------------------------------------------------------------------|
| Small (< 0.05)  | Evidence that the new data is adversely worse than the reference      |
| Large (≥ 0.05)  | Not enough evidence that the new data is worse                        |

Here, p = 0.1215 is large. We do not have sufficient evidence to conclude that Bowl is meaningfully worse than Armanaleg.

## Optional: Bayesian evidence

Bayesian evidence is optional. It provides a second summary of uncertainty alongside the standard p-value.

```python
from samesame import advanced
from samesame.bayes_factors import as_pvalue

bayes_harm = advanced.test_adverse_shift(
    reference=armanaleg,
    candidate=bowl,
    direction="higher-is-better",
    bayesian=True,
)

print(f"Bayesian p-value: {as_pvalue(bayes_harm.bayes_factor):.4f}")
```

In this example, the Bayesian summary leads to the same practical conclusion as the ordinary
p-value: there is no clear evidence that the new treatment is worse.

Use the primary API when you only need the standard p-value. Opt into `advanced.test_adverse_shift(...)`
when you need posterior draws or Bayes factors.

## Tips

- **Direction matters:** Set `direction="higher-is-worse"` when large values are harmful.
  Set `direction="higher-is-better"` when large values mean confidence or health.
- **No labels needed:** These scores can come from your existing model's predictions —
  you do not need ground truth labels. This is especially useful in production monitoring.
- **Pair with shift testing:** Run `test_shift(...)` first to detect *any* change, then run `test_adverse_shift(...)` to decide
  whether the change is harmful. See the [credit risk how-to](/examples/credit/monitor-credit-risk.md) for a full
  demonstration of both tests together.
