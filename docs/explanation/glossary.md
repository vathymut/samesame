# Glossary

Core terms — one page, one definition each. Link here on first use.

| Term | One line |
|------|----------|
| [Source / Target](#source-and-target) | Reference vs evaluation distribution (`source=` / `target=`) |
| [Score](#score) | One number per observation — risk, error, or outlier score |
| [`worse`](#worse) | Which direction is harmful (`higher` / `lower`) |
| [Harmful shift](#harmful-shift) | Target excess mass in the harmful tail |
| [Domain probability](#domain-probability) | `P(target|x)` for weighting |
| [Common support](#common-support) | Overlap region where weighting focuses the test |
| [Importance weights](#importance-weights) | `ImportanceWeights` normalized per group |
| [Reweight](#reweight) | Which group(s) to adjust (`source`/`target`/`both`) |
| [Shrinkage](#shrinkage) | `λ` blending plain ratio → uniform (default `0.5`) |
| [ESS](#effective-sample-size-ess) | Kish `(sum w)²/sum w²` — weight diagnostics |
| [Permutation test](#permutation-test) | Labels permuted, scores fixed |
| [Honest scores](#honest-out-of-sample-scores) | Out-of-sample via CV / OOB / held-out |
| [Statistic vs p-value](#statistic-vs-p-value) | `.pvalue` first, `.statistic` second |

## Source and target

**Source** — reference (training or past batch). **Target** — evaluation (production or new batch). Both are one score per observation (not feature tables). Use `source=` / `target=` everywhere.

Exchangeability under the null: source and target are labelled draws from the same mixture; permutation tests the label.

## Score

One scalar per observation. `samesame` never sees `X`, only the score you built.

| Family | Example | `worse` | Needs labels? |
|--------|---------|---------|---------------|
| Predicted risk | `P(default|x)` | `higher` | no |
| Prediction error | Brier `(y-p)²`, log-loss | `higher` | yes |
| Outlier score | `LogitGap` confidence | `lower` / `higher` | no |

Package term is **outlier score** (not anomaly/OOD). Domain probability `P(target|x)` can be the score for `test_shift`, but don't reuse it as the harm score when you also weight.

## `worse`

Polarity for `test_harmful_shift` (string or `ss.Worse`, interchangeable; enum gives autocomplete).

| Signal | `worse` |
|--------|---------|
| Predicted risk, error, atypicality | `higher` |
| Confidence / accuracy (`LogitGap`) | `lower` |

Internally `polarity = scores if worse == "higher" else -scores` so larger always means worse. Picking the wrong direction hides the harmful tail — when unsure, start with `test_shift`.

## Harmful shift

Directional change: target piles mass where source rarely goes (low `FPR` in ROC terms). Distinct from *any* shift (`test_shift`, two-sided AUC). Harm statistic `∫ TPR·(1-FPR)² dFPR` grows when that tail is heavy; mass on the beneficial side inflates AUC but not harm.

Not "average got worse" — it's a tail property anchored to source support. See [Why harm ≠ AUC](harmful-shift-statistic.md).

## Domain probability

`P(target|x)` from a domain classifier. Passed as **two** arrays to `domain_weights`:

```python
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both")
```

Prior ratio `n_source/n_target` is inferred from lengths. Clipped to `[1e-6, 1-1e-6]` before ratios. Use out-of-sample predictions; for weighting, not a harm signal.

## Common support

Region where both groups have non-negligible density.

- High overlap → unweighted test is fine.
- Low overlap → points where the other group almost never goes can dominate; weighting reframes to mutual support.

`reweight="source"` down-weights source outliers; `"target"` for target; `"both"` for mutual. Unweighted compares full populations; weighted compares overlap. Check [ESS](#effective-sample-size-ess) after weighting.

## Importance weights

`ImportanceWeights` — frozen dataclass with `.source` / `.target`, each normalized to sum to its group size. Build via `ss.domain_weights(...)` (RIW) or directly:

```python
weights = ss.ImportanceWeights(source=src_w, target=tgt_w)
weights = ss.domain_weights(source=source_prob, target=target_prob, reweight="both", shrinkage=0.5)
ess = weights.effective_sample_size()
```

Inactive groups get `1` (uniform). Weights permute with labels under the null.

## Reweight

Policy for `domain_weights(..., reweight=...)` — string or `ss.ReweightMode`.

| Mode | What it does | Use when |
|------|--------------|----------|
| `source` | reweights source toward target | source has low-overlap points |
| `target` | reweights target toward source | target has low-overlap points |
| `both` (default) | reweights both toward mutual support | both have low overlap |

Inactive groups stay `1`. Start unweighted, then compare weighted.

## Shrinkage

`shrinkage` λ in `[0,1]` — RIW (Yamada et al. 2013) trading correction strength against stability.

| `shrinkage` | Effect |
|-------------|--------|
| `0.0` | Plain ratio. Strongest, highest variance. |
| `0.5` | Default. Balanced. |
| `1.0` | Uniform. No correction. |

RIW with `r̂ = p̂/(1-p̂)·n_source/n_target`:

- Source: `w = r̂ / ((1-λ)+λr̂)`
- Target: `w = 1 / (λ+(1-λ)r̂)`

You don't compute this — `ss.domain_weights(..., shrinkage=λ)` does. If ESS collapses at `0`, raise `λ`.

## Effective sample size (ESS)

Kish's ESS per group: `ESS = (sum w)² / sum w²`.

- `ESS = n` for uniform; `→1` when one point dominates.
- Rule of thumb: worry when `ESS < n/4` (not a hard cutoff).

```python
ess = weights.effective_sample_size()
print(f"source {ess.source:.1f}/{len(weights.source)} target {ess.target:.1f}/{len(weights.target)}")
```

Compare ESS to `n` within each group, not across groups. Significant with healthy ESS is convincing; with `ESS≈1` it may be one point. If ESS stays low at `0.5`, groups barely overlap — skip weighting.

## Permutation test

Label-permutation null for both tests; scores and weights stay fixed.

- **Null:** exchangeability — source and target share the same distribution, so labels carry no information.
- **Procedure:** permute labels `n_resamples` times, recompute statistic each time.
- **p-value:** fraction at least as extreme, with +1 smoothing (Phipson & Smyth) in `(0, 1]`.

`test_shift` — two-sided on AUC (doubling capped at 1). `test_harmful_shift` — one-sided `greater` on `∫ TPR·(1-FPR)² dFPR`.

--8<-- "snippets/n-resamples.txt"

Tip: compare `result.statistic` to `result.null_distribution`. For AUC, `0.5` is chance; for harm, compare to null median.

## Honest (out-of-sample) scores

If scores come from a fitted model, make them **out-of-sample** — `cross_val_predict`, `oob_decision_function_`, or held-out. In-sample creates false separation and invalidates the test.

```python
from sklearn.model_selection import cross_val_predict
scores = cross_val_predict(model, X, y, cv=10, method="predict_proba")[:, 1]  # any estimator
# or for bagged forests:
rf.fit(X_train, y_train)
train_scores = rf.oob_decision_function_[:, 1]          # honest
deploy_scores = rf.predict_proba(X_deployment)[:, 1]    # held-out is honest
```

`samesame` only sees scores, so it cannot check this for you.

## Statistic vs p-value

Each result has `.statistic` (magnitude) and `.pvalue` (evidence).

- `.pvalue` — small (≤0.05) is evidence against the null. Two-sided for `test_shift`, one-sided `greater` for harm.
- `.statistic` — for `test_shift` it's AUC (`0.5` = chance); for harm it's `∫ TPR·(1-FPR)² dFPR` — no fixed scale, compare to null median.

Read `.pvalue` first, `.statistic` second. Both use +1 smoothing and lie in `(0,1]`. Always report both.

## References

- Yamada et al. (2013). Relative density-ratio estimation. *Neural Computation*.
- Shimodaira (2000). Improving predictive inference under covariate shift. *JSPI*.
- Kamulete et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
- Kish (1965). *Survey Sampling*. (ESS).
- Phipson & Smyth (2010). Permutation p-values.
