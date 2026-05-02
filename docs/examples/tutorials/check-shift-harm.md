# Tutorial: Check whether a shift is harmful

This tutorial is a guided first run of `test_adverse_shift(...)`.
You will start from a score where larger values mean better outcomes, set direction correctly,
and interpret whether the target sample is meaningfully worse.

**By the end, you will be able to:**

- Distinguish detecting any shift from detecting a *harmful* shift
- Use `test_adverse_shift(...)` and set direction correctly
- Read the p-value and decide whether the shift matters

In practice, this test complements `test_shift(...)`: first detect change, then assess harm.

| Test  | Question it answers                          | When to use it                              |
|-------|----------------------------------------------|---------------------------------------------|
| `test_shift` | Are the two distributions different?         | Any time you want to detect *any* change |
| `test_adverse_shift` | Is the target data *worse* than the source?  | When you care about *harmful* shifts only |

Like the shift test, this procedure works on one score per sample rather than the full table.

## What kind of input does this test need?

`test_adverse_shift` needs one score per sample that encodes how adverse that sample is.
Larger values must correspond to "worse". Examples:

- A model's predicted probability of failure or default
- An anomaly level from a detector
- A patient's discomfort or risk measure

The test asks whether the new dataset contains a disproportionate share of the large values.

## Example: comparing two treatments

This example compares two treatments for relief from leg discomfort: *Armanaleg* (the source
treatment) and *Bowl* (the target treatment). The scores measure discomfort — higher means more
discomfort, which is worse.

We want to know: **is the Bowl treatment meaningfully worse than Armanaleg?**

### Step 1 — Load the data

Relief scores are converted to discomfort scores internally, via `direction="higher-is-better"`.

```python
import numpy as np
from samesame import test_adverse_shift

relief = np.array([
     9, 14, 13,  8, 10,  5, 11,  9, 12, 10,  9, 11,  8, 11,
     4,  8, 11, 16, 12, 10,  9, 10, 13, 12, 11, 13,  9,  4,
     7, 14,  8,  4, 10, 11,  7,  7, 13,  8,  8, 13, 10,  9,
    12,  9, 11, 10, 12,  7,  8,  5, 10,  7, 13, 12, 13, 11,
     7, 12, 10, 11, 10,  8,  6,  9, 11,  8,  5, 11, 10,  8,
])
armanaleg = relief[:28]   # source treatment
bowl      = relief[28:]   # target treatment
```

### Step 2 — Run the adverse-shift test

Pass the source sample first, then the target sample. `test_adverse_shift(...)` tests whether `bowl`
contains disproportionately more high-discomfort (worse) cases than `armanaleg`:

```python
harm = test_adverse_shift(
    source=armanaleg,
    target=bowl,
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
| Small (< 0.05)  | Evidence that the target data is adversely worse than the source      |
| Large (≥ 0.05)  | Not enough evidence that the target data is worse                     |

Here, p = 0.1215 is large. We do not have sufficient evidence to conclude that Bowl is meaningfully worse than Armanaleg.

> **Advanced:** For Bayesian evidence alongside the standard p-value, see [`samesame.advanced.test_adverse_shift`](/api/advanced.md) and `samesame.bayes_factors`.

## Tips

- **Direction matters:** Set `direction="higher-is-worse"` when large values are harmful.
  Set `direction="higher-is-better"` when large values mean confidence or health.
- **No labels needed:** These scores can come from your existing model's predictions —
  you do not need ground truth labels. This is especially useful in production monitoring.
- **Pair with shift testing:** Run `test_shift(...)` first to detect *any* change, then run `test_adverse_shift(...)` to decide
  whether the change is harmful. See the [credit risk how-to](/examples/credit/monitor-credit-risk.md) for a full
  demonstration of both tests together.
