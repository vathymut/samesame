# Shift testing

Use this page when you already have a numeric signal for a source group and a target group.

## Choose the function

| Function | What it answers | Use it when |
|----------|------------------|-------------|
| `shift.detect_shift(...)` | Did anything change? | you want to detect any difference between source and target |
| `shift.detect_harm(...)` | Did the target group move in a worse direction? | you know what "worse" means for your signal |
| `shift.detect_harm_bayesian(...)` | Is the harm evidence-backed? | you want posterior draws and a Bayes factor alongside the p-value |

Examples of useful signals include predicted risk, prediction error, model confidence, and domain
classifier probabilities.

## Common controls

All functions accept:

- `n_resamples` to control the number of permutation resamples
- `batch` to limit memory use during the permutation test
- `random_state` for reproducibility
- `weights` for weighted testing with `ImportanceWeights`

`shift.detect_harm(...)` and `shift.detect_harm_bayesian(...)` also require `direction`, which must
be one of:

- `"higher-is-worse"`
- `"higher-is-better"`

## What you get back

- `shift.detect_shift(...)` returns `ShiftResult`
- `shift.detect_harm(...)` returns `HarmResult`
- `shift.detect_harm_bayesian(...)` returns `BayesianHarmResult`

In each case, the fields most users look at first are:

- `.statistic`
- `.pvalue`

`ShiftResult` also includes `.statistic_name`.
`HarmResult` and `BayesianHarmResult` include `.direction`.
All results include `.null_distribution` when you need the full permutation output.

## Posterior evidence for harmful shift

If you want posterior draws and a Bayes factor alongside the p-value, use
`shift.detect_harm_bayesian(...)` instead of `shift.detect_harm(...)`.

```python
import samesame as ss

result = ss.shift.detect_harm_bayesian(
    source_scores,
    target_scores,
    direction="higher-is-worse",
)

print(f"p-value:      {result.pvalue:.4f}")
print(f"Bayes factor: {result.bayes_factor:.2f}")
```

`BayesianHarmResult` adds `.posterior` (the posterior draws) and `.bayes_factor`. The optional
`threshold` parameter sets the statistic value above which evidence counts as harm; it defaults to
`1 / 12`.

## API

::: samesame.shift.detect_shift

::: samesame.shift.detect_harm

::: samesame.shift.detect_harm_bayesian

## Result types

::: samesame.shift.TestResult

::: samesame.shift.ShiftResult

::: samesame.shift.HarmResult

::: samesame.shift.BayesianHarmResult
