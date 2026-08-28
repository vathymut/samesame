# Tutorial: Test whether the shift is harmful

Is a shift harmful — not just different? In 5 minutes you will:

- distinguish "different" from "worse"
- choose the correct `worse` direction
- run `ss.test_harmful_shift` and read the result

`ss.test_shift` asks whether [source and target](../../explanation/glossary.md#source-and-target) differ at all (two-sided, AUC). `ss.test_harmful_shift` asks the narrower question: did target shift toward the harmful tail (one-sided)? See [Glossary: `worse`](../../explanation/glossary.md#worse).

!!! info "Prerequisites"
    - [Detect any distributional shift](detect-distribution-shift.md) — shift test and honest scores.

## What you need

- source and target scores (one number per observation)
- a decision about which direction is worse

## Step 1 — Make an example

Imagine these are confidence scores, where **higher is better**.

```python
import numpy as np

rng = np.random.default_rng(12345)

source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
target_quality = rng.normal(loc=0.72, scale=0.07, size=400)
```

Target is slightly lower — a harmful shift for a higher-is-better signal.

## Step 2 — Compare any change vs harmful change

=== "Higher is better (confidence, accuracy)"

    ```python
    import samesame as ss

    shift = ss.test_shift(source=source_quality, target=target_quality, rng=rng)

    harm = ss.test_harmful_shift(
        source=source_quality,
        target=target_quality,
        worse="lower",  # smaller = worse for confidence (or ss.Worse.LOWER)
        rng=rng,
    )

    print(f"Shift p-value: {shift.pvalue:.4f}")
    print(f"Harm  p-value: {harm.pvalue:.4f}")
    ```

    `worse="lower"` means a shift toward *smaller* scores is harmful. Scores are negated internally so larger always means worse.

=== "Higher is worse (risk, error, atypicality)"

    ```python
    import samesame as ss

    rng = np.random.default_rng(12345)
    source_risk = rng.normal(loc=0.20, scale=0.07, size=400)
    target_risk = rng.normal(loc=0.28, scale=0.07, size=400)

    harm = ss.test_harmful_shift(source=source_risk, target=target_risk, worse="higher", rng=rng)
    print(f"Harm p-value: {harm.pvalue:.4f}")
    ```

    `worse="higher"` (or `ss.Worse.HIGHER`) means larger scores are worse — for risk, error, and atypical outlier scores.

## How to read the result

--8<-- "snippets/pvalue-guidance.txt"

- Small `test_shift` p-value — groups differ.
- Small `test_harmful_shift` p-value — target also shifted toward the harmful direction you declared.

--8<-- "snippets/worse-table.txt"

--8<-- "snippets/honest-scores-ref.txt"

For the statistic behind the test, see [Why the harm statistic is not just AUC](../../explanation/harmful-shift-statistic.md). For the API, see [Shift testing](../../api/testing.md). When feature support differs, continue to [Adjust for covariate shift with importance weights](adjust-for-covariate-shift.md).
