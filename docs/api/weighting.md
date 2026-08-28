# Importance weights

Use `samesame.weights` when a plain comparison gives too much influence to observations the other group rarely contains. Weighting changes the question: you compare groups mainly on **common support**.

## Choose an approach

| Situation | What to do |
|-----------|------------|
| No overlap concern | Omit `weights` |
| You already have sample weights | Wrap them in `ss.ImportanceWeights(source=..., target=...)` |
| You have domain probabilities `P(target \| x)` | Build weights with `ss.domain_weights` |

```python
import samesame as ss

# no weighting
result = ss.test_shift(source=source_scores, target=target_scores, rng=12345)

# custom weights
result = ss.test_shift(
    source=source_scores,
    target=target_scores,
    weights=ss.ImportanceWeights(source=source_weights, target=target_weights),
    rng=12345,
)

# from a domain classifier
weights = ss.domain_weights(
    source=source_domain_prob,  # P(target | x) for source observations
    target=target_domain_prob,  # P(target | x) for target observations
    reweight="both",  # or ss.ReweightMode.BOTH
    shrinkage=0.5,
)

result = ss.test_harmful_shift(
    source=source_scores, target=target_scores, worse="higher", weights=weights, rng=12345,
)
```

--8<-- "snippets/honest-scores.txt"

--8<-- "snippets/clipping-note.txt"

## When `ss.domain_weights` helps

Use it when source and target don't overlap well and you want the comparison to emphasize common support.

It takes:

- `source` and `target` — domain probabilities `P(target | x)` as **separate** 1-D arrays in `[0, 1]`, each aligned to the corresponding score array. The prior ratio `n_source / n_target` is inferred from lengths.
- `reweight` — whether to reweight `source`, `target`, or `both` (default `"both"`). Accepts plain string or `ss.ReweightMode` (`"source"` / `ss.ReweightMode.SOURCE`).
- `shrinkage` (λ) in `[0, 1]` — RIW shrinkage trading correction strength against stability. `0` = plain ratio, `1` = uniform. Default `0.5`.

Probabilities at `0` or `1` are clipped to `[1e-6, 1 − 1e-6]` before weighting (see note above).

## Choosing what to reweight

| Mode | What it emphasizes |
|------|--------------------|
| `reweight="source"` (`ss.ReweightMode.SOURCE`) | overlap from the source side; target unchanged |
| `reweight="target"` (`ss.ReweightMode.TARGET`) | overlap from the target side; source unchanged |
| `reweight="both"` (`ss.ReweightMode.BOTH`) | common support from both sides (default) |

Inactive groups get weight `1` for every observation, then normalized — so they are uniform. Start with `shrinkage=0.5`; lower values are stronger but higher variance. Check [ESS](../examples/weighting/diagnose-weight-concentration.md) before lowering.

## Effective sample size

Once you have `ImportanceWeights`, call `.effective_sample_size()` for Kish's ESS per group. ESS ≤ `n`; it drops toward `1` when a few points hog the weight.

```python
ess = weights.effective_sample_size()
print(ess.source, ess.target)  # compare each to its n
```

Rule of thumb: worry when `ESS < n/4`. A significant weighted result with healthy ESS is convincing; with `ESS ≈ 1` it may be driven by one or two observations.

For a worked sweep, see [Diagnose weight concentration](../examples/weighting/diagnose-weight-concentration.md). For the intuition, see [When importance weights help](../explanation/importance-weights-rationale.md). For the full tutorial, see [Adjust for covariate shift with importance weights](../examples/tutorials/adjust-for-covariate-shift.md).

## API

::: samesame.weights.domain_weights

::: samesame.weights.ImportanceWeights

::: samesame.weights.ReweightMode

::: samesame.weights.EffectiveSampleSize
