# Shift testing

One interpretable score `ϕ(x)` per observation, typically an outlier score such as severity or harm. Compare it between **source** (reference) and **target** (current deployment). Larger means worse. Choose `ϕ` to encode the notion of *worse* you care about.

## Scope

Reference for the two permutation tests. For choosing `ϕ` and `worse`, see [Core concepts](../explanation/core-concepts.md) and [How the harm test works](../explanation/harmful-shift-statistic.md). For weighting, see [Importance weights](weighting.md). Hands-on: [Get started](../examples/tutorials/get-started.md).

## Which test?

| Function | Question | When to use |
|----------|----------|-------------|
| `ss.test_shift` | Do source and target differ? | Any change matters — screen first, then ask about harm |
| `ss.test_harmful_shift` | Did the target move toward the harmful tail you specified? | You can declare `worse` in advance and care about one tail |

`ss.test_shift(source, target, *, weights=None, n_resamples=9999, rng=None)`: permutation test for any shift. Pass one score `ϕ(x)` per observation (see [Core concepts](../explanation/core-concepts.md)). Returns AUC in `.statistic` (`0.5` is no separation) and a two-sided p-value (doubles smaller tail, `+1` smoothing). Example: `ss.test_shift(source=scores_ref, target=scores_cur, rng=rng)`. Details below under [Reading results](#reading-results).

`ss.test_harmful_shift(source, target, *, worse, weights=None, n_resamples=9999, rng=None)`: permutation test for tail harm. Same `ϕ(x)` plus `worse="higher"|"lower"` (or `ss.Worse`) declared before looking. Returns weighted AUC and one-sided `greater` p-value. Example: `ss.test_harmful_shift(source=risk_ref, target=risk_cur, worse="higher", rng=rng)`. It reads tail mass where source is rare; see [How the harm test works](../explanation/harmful-shift-statistic.md).

--8<-- "snippets/worse-declaration.txt"

--8<-- "snippets/worse-table.txt"

Concepts and full definitions: [Core concepts](../explanation/core-concepts.md).

## Reading results

Both return `.statistic`, `.pvalue`, and `.null_distribution` (permuted labels, scores fixed).

--8<-- "snippets/pvalue-caveat.txt"

Tails: `test_shift` is two-sided (doubles smaller tail, capped at 1); `test_harmful_shift` is one-sided `greater`. Both use `+1` smoothing (Phipson & Smyth, 2010).

- `test_shift`: `.statistic` is AUC (`0.5` is no separation; near `0` or `1` is strong).
- `test_harmful_shift`: `.statistic` is weighted AUC; read against `null_distribution` and your score's scale (see [How the harm test works](../explanation/harmful-shift-statistic.md)); records `.worse`.

### Honest scores {#honest-scores}

Valid p-values need out-of-sample scores. `samesame` only sees what you pass in.

--8<-- "snippets/honest-scores.txt"

Never fit and test on the same rows (Kamulete 2022 §5):

- **Permutation (default)**: fit `ϕ` once, permute labels (`n_resamples=9999`).
- **Out-of-bag**: `oob_decision_function_` for bagged models (`examples/credit/monitor-credit.md`).
- **Cross-fit**: `cross_val_predict` or any k-fold scoring each row out-of-sample (`examples/tutorials/get-started.md`).

??? tip "Reproducibility"
    Pass `rng=np.random.default_rng(12345)` (`n_resamples=9999`; `999` while exploring, `19999` below `0.001`).

??? details "Source files"
    `src/samesame/shift.py` · `src/samesame/_permutation.py` · `src/samesame/_statistics.py`

## API

::: samesame.shift.test_shift

::: samesame.shift.test_harmful_shift

## Result types

::: samesame.shift.ShiftResult

::: samesame.shift.HarmfulShiftResult

::: samesame.shift.Worse
