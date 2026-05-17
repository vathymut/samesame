# Shift detection

Use this page for the two primary user-facing calls:
`shift.detect_shift(...)` and `shift.detect_harm(...)`.
Start here if you are new to the package or want the simplest API surface.

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult` with `.statistic`, `.pvalue`, `.statistic_name`, and `.null_distribution`
- `shift.detect_harm(...)` returns `HarmResult` with `.statistic`, `.pvalue`, `.direction`, and `.null_distribution`
- `shift.detect_harm(..., include_posterior=True)` additionally returns `.posterior` and `.bayes_factor`

## Additional controls

Both functions accept direct keyword arguments for resampling, weighting, and reproducibility.

- `n_resamples` controls the number of permutation resamples. When `include_posterior=True`, the same value also controls the number of posterior draws.
- `batch` caps peak memory for the permutation path. It applies to the permutation result only.
- `random_state` accepts a seed or NumPy RNG for reproducibility.
- `weights` accepts `ImportanceWeights` for weighted shift or harmful-shift testing.

## Advanced harmful-shift evidence

Use `include_posterior=True` when you want posterior draws and a Bayes factor alongside the standard harmful-shift p-value.

```python
import samesame as ss

result = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
    include_posterior=True,
)

print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {result.bayes_factor:.2f}")
```

`threshold` sets the effect-size threshold used for the Bayes factor. It is only valid when `include_posterior=True`; otherwise `detect_harm(...)` raises a `ValueError`. When posterior evidence is not requested, `HarmResult.posterior` and `HarmResult.bayes_factor` are `None`.

## Functions

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

## Return types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult

::: samesame.shift.TestResult
