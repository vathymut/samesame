# Glossary

Core terms used across `samesame`. Keep this as the single source for wording; link here on first use in tutorials and how-to guides.

## Source and target

**Source** — reference distribution. Usually training data or a past batch. Scores from source are the baseline.

**Target** — evaluation distribution. Usually production data or a new batch. Scores from target are tested for change.

Both are represented by one numeric score per observation (not raw feature tables). Use `source=` and `target=` arguments consistently; `group`/`membership_prob` are historical names (see `CONTEXT.md`).

## Score

A scalar signal derived from a model or raw measurement. Three families `samesame` is built for:

- **Predicted risk** — e.g., `P(default | x)`. Business impact directly.
- **Prediction error** — e.g., Brier score, log-loss. Needs labels.
- **Outlier score** — e.g., `LogitGap` confidence. Higher = more typical / more certain; lower = more atypical. Package term is *outlier score* (not "anomaly score" or "OOD score").

Any scalar works; `samesame` only tests it.

## `worse`

Polarity parameter that defines harmful direction. Declared once per `test_harmful_shift` call:

- `worse="higher"` (or `ss.Worse.HIGHER`) — larger scores mean harm (risk, error, atypicality).
- `worse="lower"` (or `ss.Worse.LOWER`) — smaller scores mean harm (confidence, accuracy).

Internally scores are transformed so larger always means worse (`polarity = scores if worse == "higher" else -scores`). See [Shift testing](../api/testing.md).

## Harmful shift

A directional distributional change where target carries excess mass in the harmful tail you declared. Detected via `shift.test_harmful_shift(source_scores, target_scores, worse=...)`. Distinct from *any* shift (two-sided `test_shift`).

## Domain probability

`P(target | x)` — output of a domain classifier trained to distinguish source from target. Passed as **separate** 1-D arrays `source` and `target` to `domain_weights(...)`. The prior ratio `n_source / n_target` is inferred from lengths, not tuned.

Used only for **weighting**, not as a harm signal. Don't reuse `P(target | x)` as the score you test for harm when you also weight.

## Common support

Region of feature space where both source and target have non-negligible density. **Low-overlap** observations sit where the other group almost never goes and can dominate an unweighted test. Weighting reframes the question to common support.

## Importance weights

`ImportanceWeights` — frozen dataclass with `.source` and `.target` arrays, normalized so each group sums to its size. Built via `domain_weights(...)` from domain probabilities or constructed directly from custom weights. See [Importance weights](../api/weighting.md).

## Reweight

Policy for `domain_weights(..., reweight=...)`:

- `"source"` (`ss.ReweightMode.SOURCE`) — reweight source toward target; target unchanged.
- `"target"` (`ss.ReweightMode.TARGET`) — reweight target toward source; source unchanged.
- `"both"` (`ss.ReweightMode.BOTH`, default) — reweight both toward mutual support.

Inactive groups get weight `1` (uniform after normalization).

## Shrinkage

`shrinkage` (λ) in `[0, 1]` — RIW (Relative Importance Weight) shrinkage trading correction strength against stability (Yamada et al. 2013). `0` = plain density ratio (strongest, highest variance); `1` = uniform (no correction); `0.5` = default balance. See [When importance weights help](importance-weights-rationale.md).

## Effective sample size (ESS)

Kish's ESS per group: `(sum w)² / sum w²`. `ESS = n` for uniform weights; `→ 1` when one point dominates. Call `weights.effective_sample_size()` — it returns `.source` and `.target`.

--8<-- "snippets/ess-rule.txt"

See [Diagnose weight concentration](../examples/weighting/diagnose-weight-concentration.md) and `EffectiveSampleSize`.

## Permutation test

Label-permutation null. Scores (and weights, if given) stay fixed; only labels are permuted `n_resamples` times.

- `test_shift` — two-sided on ROC AUC; doubling capped at `1`.
- `test_harmful_shift` — one-sided `greater` on the harm statistic.

p-values use +1 smoothing (Phipson & Smyth) and lie in `(0, 1]`.

--8<-- "snippets/n-resamples.txt"

## Honest (out-of-sample) scores

Scores from a fitted model must be **out of sample** — cross-validation, OOB, or held-out set. In-sample predictions create false separation and invalidate the test. `samesame` only sees scores, not how they were made, so it cannot check this for you.

## Statistic vs p-value

- `.pvalue` — evidence against the null. Small (typically ≤ 0.05) means unlikely under no shift.
- `.statistic` — magnitude. For `test_shift` it's ROC AUC (`0.5` = chance); for `test_harmful_shift` it has no fixed scale — compare observed vs null median (see [Why the harm statistic is not just AUC](harmful-shift-statistic.md)).

Read `.pvalue` first; `.statistic` second.

## References

- Yamada et al. (2013). Relative density-ratio estimation. *Neural Computation*.
- Shimodaira (2000). Improving predictive inference under covariate shift. *JSPI*.
- Kamulete et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
- Kish (1965). *Survey Sampling*. (ESS).
- Phipson & Smyth (2010). Permutation p-values.
