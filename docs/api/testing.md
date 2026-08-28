# Shift testing

Use this page when you already have one numeric score for each group. The tests don't operate on raw feature tables — turn those into a score first.

--8<-- "snippets/honest-scores.txt"

## Choose a function

| Function | What it answers | Use it when |
|----------|-----------------|-------------|
| `ss.test_shift` | Did anything change? | you want to detect any difference between source and target |
| `ss.test_harmful_shift` | Did target shift toward worse outcomes? | you know what "worse" means for your signal |

Useful signals: predicted risk, prediction error (Brier/log-loss), or outlier scores (e.g., `LogitGap` confidence). Domain probability `P(target | x)` is a valid signal for the *shift* test only — don't reuse it as the harm signal when you also weight.

## Common controls

All functions accept:

- `n_resamples` — permutation resamples. Default `9999`. Use `999` while exploring, `9999` for the final result, `19999` if you need `p < 0.001` resolution. Cost is `O(n log n)` per resample, `O(n)` memory.
- `rng` — reproducibility. Accepts `int` seed, `np.random.Generator`/`RandomState`, or `None`.
- `weights` — `ImportanceWeights` from `ss.domain_weights` or constructed directly. Must match `len(source)` and `len(target)`.

`ss.test_harmful_shift` also requires `worse`:

- `worse="higher"` (or `ss.Worse.HIGHER`) — larger scores mean harm (risk, error, anomaly).
- `worse="lower"` (or `ss.Worse.LOWER`) — smaller scores mean harm (confidence, accuracy).

Plain strings and `ss.Worse` are both accepted; the enum gives autocomplete and typo protection.

## What you get back

- `ss.test_shift` → `ShiftResult` (`.statistic` is ROC AUC, two-sided p-value)
- `ss.test_harmful_shift` → `HarmfulShiftResult` (`.statistic` is harm statistic, one-sided `greater` p-value, plus `.worse`)

Both include `.null_distribution` when you need the full permutation output. `.statistic` describes separation; `.pvalue` assesses evidence. p-values are in `(0, 1]` with +1 smoothing (Phipson & Smyth); two-sided doubling is capped at `1`.

??? tip "Which result to read first?"
    Start with `.pvalue` (evidence). Use `.statistic` to describe magnitude — AUC ≈ `0.5` is chance; harm statistic has no fixed scale, compare observed vs null median. See [Why the harm statistic is not just AUC](../explanation/harmful-shift-statistic.md).

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
