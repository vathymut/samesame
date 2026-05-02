# How-to: Use double-weighting for covariate-shift adaptation

**Use this guide when:** both your source and target groups contain outliers that are foreign
to the other group, and you want adverse-shift testing to focus exclusively on the region of
feature space they share.

**What you'll do:**

- Understand when source-only reweighting is insufficient
- Apply `mode="both"` to reweight source and target simultaneously
- Run a weighted adverse-shift test and compare all three weighting approaches

!!! note "Before you start"
    This guide assumes you have completed:

    - [Adjust for covariate shift with importance weights](../tutorials/adjust-for-covariate-shift.md)
    - [Use source reweighting for adverse-shift testing](source-reweighting.md)

---

## When single-group reweighting is not enough

Source reweighting (`mode="source"`) down-weights training samples that are foreign to
deployment. But if the deployment population also contains samples with feature values that
never appeared in training, those target outliers remain at unit weight and can inflate the
test statistic in the same way. Double-weighting corrects both sides simultaneously.

!!! warning
    Double-weighting is the most aggressive correction mode. Use it only when you have
    evidence that outliers are present in both groups. If the target group is well-contained
    within the source distribution, source reweighting is sufficient.

---

## Step 1 — Set up two score streams

Use the same HELOC setup as
[Use source reweighting for adverse-shift testing](source-reweighting.md).
After completing that guide, you have three variables in scope:

- `membership_prob` — OOB probabilities from `rf_domain` (for weighting)
- `bad_train` / `bad_test` — predicted default-risk scores (for adverse-shift testing)

If you are starting fresh, run the full setup from Step 1 of that guide before continuing.

---

## Step 2 — Apply double-weighting

Change `mode` from `"source"` to `"both"`. Source outliers receive lower weights via the
forward density ratio; target outliers receive lower weights via the inverse density ratio:

```python
from samesame import test_adverse_shift

double = test_adverse_shift(
    source=bad_train,
    target=bad_test,
    direction="higher-is-worse",
    membership_prob=membership_prob,
    mode="both",
    alpha_blend=0.5,
    rng=np.random.default_rng(12345),
)
print(f"Double-weighted — statistic: {double.statistic:.4f}, p-value: {double.pvalue:.4f}")
```

---

## Step 3 — Compare all three weighting modes

Run all three tests side by side to see how the statistic changes as each mode narrows focus:

```python
from samesame import test_adverse_shift
import numpy as np

unweighted = test_adverse_shift(
    source=bad_train,
    target=bad_test,
    direction="higher-is-worse",
    rng=np.random.default_rng(12345),
)
source_rw = test_adverse_shift(
    source=bad_train,
    target=bad_test,
    direction="higher-is-worse",
    membership_prob=membership_prob,
    mode="source",
    alpha_blend=0.5,
    rng=np.random.default_rng(12345),
)
print(f"Unweighted    — statistic: {unweighted.statistic:.4f}, p-value: {unweighted.pvalue:.4f}")
print(f"Source only   — statistic: {source_rw.statistic:.4f}, p-value: {source_rw.pvalue:.4f}")
print(f"Double        — statistic: {double.statistic:.4f}, p-value: {double.pvalue:.4f}")
```

| Weighting | Focus |
|-----------|-------|
| None | Full populations, all outliers included. |
| `mode="source"` | Overlap from the source side; target outliers still at unit weight. |
| `mode="both"` | Common support only — outliers in both groups down-weighted. |

The statistic changes across modes because each mode asks a slightly different question.
Double-weighting measures adverse shift restricted to common support from both sides.

---

## Choosing `alpha_blend`

The default `alpha_blend=0.5` is a balanced blend between the plain density ratio (`0.0`) and
uniform weights (`1.0`). For double-weighting:

- **Lower `alpha_blend` (e.g., 0.2):** More aggressive correction; use only with a
  well-calibrated membership classifier and a large overlap region.
- **Higher `alpha_blend` (e.g., 0.8):** More conservative; close to uniform weights.
  Use when you are uncertain about the quality of membership probabilities.

For the mathematical relationship between `alpha_blend` and weight magnitude, see
[Why importance weights stabilise shift detection](../../explanation/importance-weights-rationale.md).

---

## See also

- [Use source reweighting for adverse-shift testing](source-reweighting.md)
  — the simpler alternative when only the source has outliers.
- [Why importance weights stabilise shift detection](../../explanation/importance-weights-rationale.md)
  — RIW formulas and the three-mode decision guide.
- [Weighting strategies](../../api/weighting.md) — full `membership_prob` and `mode` API reference.
