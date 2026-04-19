# How to Test for Harmful Shift (Noninferiority)

**Goal:** Determine whether a new dataset is *meaningfully worse* than a reference dataset,
not just different.

## The problem with asking "are these the same?"

Imagine you deploy a machine learning model in January and again in February.
You run a distribution test (CTST) and get a small p-value — the February data is statistically
different from January. Should you be worried?

Not necessarily. Any two real-world datasets will differ slightly due to random variation.
CTST is sensitive enough to pick up even tiny, inconsequential differences.

What you usually *actually* want to know is: **"Is February's data worse in a way that could
hurt my model?"** That is the question DSOS answers.

## CTST vs. DSOS at a glance

| Test  | Question it answers                          | When to use it                              |
|-------|----------------------------------------------|---------------------------------------------|
| CTST  | Are the two distributions different?         | Any time you want to detect *any* change     |
| DSOS  | Is the new data *worse* than the reference?  | When you care about *harmful* shifts only   |

Use both together: CTST to detect change, DSOS to judge severity.

## What are "outlier scores"?

DSOS works by comparing **outlier scores** — a number assigned to each sample that represents
how unusual or risky it is. Higher scores mean "worse". Examples:

- A model's predicted probability of failure or default
- A reconstruction error from an anomaly detection model
- A measure of patient discomfort or risk

DSOS asks: does the new dataset have disproportionately more high-scoring (worse) samples?

## How DSOS works (in plain terms)

1. Combine both datasets and label them (reference = 0, new = 1)
2. Compute a weighted AUC that gives extra importance to the highest-scoring samples
3. Use a one-sided permutation test to ask: are the worst samples concentrated in the new dataset?

No parametric assumptions are required, and you do not need to specify a margin in advance.

## Example: comparing two treatments

This example is based on a [SAS case study](https://support.sas.com/resources/papers/proceedings15/SAS1911-2015.pdf)
comparing two treatments for relief from leg discomfort: *Armanaleg* (the established reference)
and *Bowl* (the new treatment). The scores below measure discomfort — higher means more discomfort,
which is worse.

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

### Step 2 — Run the DSOS test

Pass the reference sample first, then the new sample. DSOS tests whether `bowl`
contains disproportionately more high-discomfort (worse) cases than `armanaleg`:

```python
from samesame.nit import DSOS
from samesame.bayes import as_pvalue

dsos = DSOS.from_samples(armanaleg, bowl)

print(f"Frequentist p-value: {dsos.pvalue:.4f}")
print(f"Bayesian    p-value: {as_pvalue(dsos.bayes_factor):.4f}")
```

**Output:**

```text
Frequentist p-value: 0.1215
Bayesian    p-value: 0.1159
```

## Reading the results

| p-value         | What it means                                                         |
|-----------------|-----------------------------------------------------------------------|
| Small (< 0.05)  | Evidence that the new data is adversely worse than the reference      |
| Large (≥ 0.05)  | Insufficient evidence that the new data is worse (noninferior result) |

Here, p = 0.1215 is large. We cannot conclude that Bowl is meaningfully worse than Armanaleg —
the new treatment passes the noninferiority check.

## DSOS and WeightedAUC

`DSOS` is an alias for `WeightedAUC` in the API.

- Use `DSOS.from_samples(reference_scores, new_scores)` for the standard unweighted path.
- If you need `sample_weight`, construct `WeightedAUC(...)` directly with
  `actual`, `predicted`, and `sample_weight`.

### Frequentist vs. Bayesian p-value

DSOS provides two ways to summarise the evidence:

- **Frequentist p-value** (`dsos.pvalue`): the standard approach, based on permutations
- **Bayesian p-value** (`as_pvalue(dsos.bayes_factor)`): derived from the Bayes factor,
  which quantifies how much evidence there is *in favour of* an adverse shift

Both tell the same story here. The Bayesian option is useful when you want to make
probability statements about the hypothesis, or when you are running sequential tests
over time.

## Tips

- **Score direction matters:** Make sure high scores mean "worse". If your scores are
  confidence values (higher = better), negate them before passing to DSOS.
- **No labels needed:** Outlier scores can come from your existing model's predictions —
  you do not need ground truth labels. This is especially useful in production monitoring.
- **Pair with CTST:** Run CTST first to detect *any* change, then run DSOS to decide
  whether the change is harmful. See the [Credit example](credit-example.md) for a full
  demonstration of both tests together.
