# Tutorial: Check whether a shift is harmful

**What you'll learn:**

- The difference between detecting any shift and detecting a *harmful* shift
- How to use `test_adverse_shift(...)` and tell it which direction means worse
- How to read the p-value and decide whether the shift matters

**Goal:** Determine whether a new dataset is *meaningfully worse* than a reference dataset,
not just different.

## The problem with asking "are these the same?"

Imagine you deploy a machine learning model in January and again in February.
You run `test_shift(...)` and get a small p-value — the February data is statistically
different from January. Should you be worried?

Not necessarily. Any two real-world datasets will differ slightly due to random variation.
Shift testing is sensitive enough to pick up even tiny, inconsequential differences.

What you usually *actually* want to know is: **"Is February's data worse in a way that could
hurt my model?"** That is the question `test_adverse_shift(...)` answers.

## Shift vs. Adverse Shift at a glance

| Test  | Question it answers                          | When to use it                              |
|-------|----------------------------------------------|---------------------------------------------|
| `test_shift` | Are the two distributions different?         | Any time you want to detect *any* change |
| `test_adverse_shift` | Is the new data *worse* than the reference?  | When you care about *harmful* shifts only |

Use both together: `test_shift` to detect change, `test_adverse_shift` to judge severity.

Like the shift test, this works on one number per sample rather than the full table.

## What kind of input does this test need?

`test_adverse_shift` needs one number per sample that says how concerning that sample is.
Larger values must line up with "worse". Examples:

- A model's predicted probability of failure or default
- An anomaly level from a detector
- A patient's discomfort or risk measure

The test asks: does the new dataset contain too many of the large values?

## How the test works

1. Pool both datasets and mark which samples belong to each group.
2. Check whether the largest values are concentrated in the new dataset.
3. Shuffle the group labels many times to build a baseline. If the real concentration
  is unusually high compared to those shuffled baselines, the test flags it as significant
   (small p-value).

You do not need to assume a particular distribution shape, and you do not need to set a cutoff in advance.

## Example: comparing two treatments

This example compares two treatments for relief from leg discomfort: *Armanaleg* (the established
reference) and *Bowl* (the new treatment). The scores measure discomfort — higher means more
discomfort, which is worse.

We want to know: **is the Bowl treatment meaningfully worse than Armanaleg?**

### Step 1 — Load the data

Relief scores are converted to discomfort scores by flipping them, so that higher always means worse:

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
discomfort = [max(relief) - s for s in relief]  # flip: higher = more discomfort = worse

armanaleg = np.array(discomfort[:28])   # reference treatment
bowl = np.array(discomfort[28:])        # new treatment
```

### Step 2 — Run the adverse-shift test

Pass the reference sample first, then the new sample. `test_adverse_shift(...)` tests whether `bowl`
contains disproportionately more high-discomfort (worse) cases than `armanaleg`:

```python
from samesame import test_adverse_shift

harm = test_adverse_shift(
  reference=armanaleg,
  candidate=bowl,
  direction="higher-is-worse",
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

Here, p = 0.1215 is large. We cannot conclude that Bowl is meaningfully worse than Armanaleg —
the new treatment is acceptable.

## Optional: Bayesian evidence

Bayesian evidence is optional. Think of it as a second way to summarise uncertainty
alongside the standard p-value.

```python
from samesame import advanced
from samesame.bayes_factors import as_pvalue

detail = advanced.test_adverse_shift(
    reference=armanaleg,
    candidate=bowl,
    direction="higher-is-worse",
    bayesian=True,
)

print(f"Bayesian p-value: {as_pvalue(detail.bayes_factor):.4f}")
```

In this example, the Bayesian summary leads to the same practical takeaway as the ordinary
p-value: there is no clear evidence that the new treatment is worse.

Use the primary API when you only need the standard p-value. Opt into `advanced.test_adverse_shift(...)`
when you need posterior draws or Bayes factors.

## Tips

- **Direction matters:** Set `direction="higher-is-worse"` when large values are harmful.
  Set `direction="higher-is-better"` when large values mean confidence or health.
- **No labels needed:** These per-row numbers can come from your existing model's predictions —
  you do not need ground truth labels. This is especially useful in production monitoring.
- **Pair with shift testing:** Run `test_shift(...)` first to detect *any* change, then run `test_adverse_shift(...)` to decide
  whether the change is harmful. See the [credit risk how-to](credit-example.md) for a full
  demonstration of both tests together.
