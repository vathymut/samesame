# Tutorial: Is the shift harmful?

Is target *worse*, not just different?

- `test_shift` — did they differ at all? (two-sided, AUC)
- `test_harmful_shift` — did target shift toward the harmful tail? (one-sided, needs `worse`)

Prerequisite: [Detect any shift](detect-distribution-shift.md).

## Make an example

Confidence where **higher is better**:

```python
import numpy as np
rng = np.random.default_rng(12345)
source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
target_quality = rng.normal(loc=0.72, scale=0.07, size=400)  # slightly lower — harmful
```

## Compare any change vs harmful change

=== "Higher is better (confidence, accuracy)"

    ```python
    import samesame as ss
    shift = ss.test_shift(source=source_quality, target=target_quality, rng=rng)
    harm = ss.test_harmful_shift(source=source_quality, target=target_quality, worse="lower", rng=rng)
    print(f"Shift p={shift.pvalue:.4f} Harm p={harm.pvalue:.4f}")  # → 0.0001, 0.0001
    ```
    `worse="lower"` — shift toward *smaller* scores is harmful. Strings and `ss.Worse` are interchangeable; internally scores are negated so larger always means worse.

=== "Higher is worse (risk, error)"

    ```python
    import samesame as ss
    rng = np.random.default_rng(12345)
    source_risk = rng.normal(loc=0.20, scale=0.07, size=400)
    target_risk = rng.normal(loc=0.28, scale=0.07, size=400)
    harm = ss.test_harmful_shift(source=source_risk, target=target_risk, worse="higher", rng=rng)
    print(f"Harm p={harm.pvalue:.4f}")  # → 0.0001
    ```

## How to read

--8<-- "snippets/pvalue-guidance.txt"

- Small `test_shift` p — groups differ.
- Small `test_harmful_shift` p — target also shifted toward the harmful direction you declared.

| Signal | `worse` |
|--------|---------|
| Predicted risk, error, atypicality | `higher` |
| Confidence (`LogitGap`) | `lower` |

When feature support differs, see [Weight for common support](../weighting/weight-for-common-support.md). For the statistic, see [Why harm ≠ AUC](../../explanation/harmful-shift-statistic.md).
