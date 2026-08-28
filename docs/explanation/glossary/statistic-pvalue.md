# Statistic vs p-value

Each result has `.statistic` (magnitude) and `.pvalue` (evidence).

- `.pvalue` — evidence against the null. Small (typically ≤ 0.05) means unlikely under no shift. `test_shift` is two-sided; `test_harmful_shift` is one-sided `greater`.
- `.statistic` — magnitude. For `test_shift` it's ROC AUC (`0.5` = chance); for `test_harmful_shift` it's the harm statistic `∫ TPR·(1-FPR)² dFPR` — no fixed scale, compare observed vs null median.

```
Read .pvalue first; .statistic second.
```

Permutation p-values use +1 smoothing (Phipson & Smyth) and lie in `(0, 1]`; two-sided doubling is capped at `1`.

**How to read:**

- `p ≤ 0.05` — evidence against null (shift detected).
- `statistic` far from null median — practically large shift.
- Always report both; a tiny p-value with near-null statistic is statistically but not practically significant.

See [Why the harm statistic is not just AUC](../harmful-shift-statistic.md#what-the-harm-statistic-integrates) for harm scale and [Permutation test](permutation-test.md) for inference.
