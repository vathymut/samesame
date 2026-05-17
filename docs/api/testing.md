# Shift testing

Use this page when you already have a numeric signal for a source group and a target group.

## Choose the function

| Function | What it answers | Use it when |
|----------|------------------|-------------|
| `shift.detect_shift(...)` | Did anything change? | you want to detect any difference between source and target |
| `shift.detect_harm(...)` | Did the target group move in a worse direction? | you know what "worse" means for your signal |

Examples of useful signals include predicted risk, prediction error, model confidence, and domain
classifier probabilities.

## Common controls

Both functions accept:

- `n_resamples` to control the number of permutation resamples
- `batch` to limit memory use during the permutation test
- `random_state` for reproducibility
- `weights` for weighted testing with `ImportanceWeights`

`shift.detect_harm(...)` also requires `direction`, which must be one of:

- `"higher-is-worse"`
- `"higher-is-better"`

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult`
- `shift.detect_harm(...)` returns `HarmResult`

In both cases, the fields most users look at first are:

- `.statistic`
- `.pvalue`

`ShiftResult` also includes `.statistic_name`.
`HarmResult` also includes `.direction`.
Both results include `.null_distribution` when you need the full permutation output.

## Posterior evidence for harmful shift

If you want posterior draws and a Bayes factor alongside the p-value, set
`include_posterior=True`.

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

`threshold` is only valid when `include_posterior=True`. Otherwise `detect_harm(...)` raises a
`ValueError`.

## API

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult

::: samesame.shift.TestResult
