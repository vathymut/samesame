# Glossary

Core terms used across `samesame`. Single source for wording — link here on first use in tutorials and how-to guides.

!!! tip "How to use this page"
    **Novice:** Core concepts in order — Source → Score → `worse` → Harmful shift → Honest scores → Statistic vs p-value. **Expert:** jump to Weighting (Shrinkage, ESS, Importance weights) or Inference (Permutation test). Groups match the sidebar; each term links to a focused page.

| Term | One line | Details |
|------|----------|---------|
| [Source / Target](#source-and-target) | Reference vs evaluation distribution (`source=` / `target=`) | [→ page](glossary/source-target.md) |
| [Score](#score) | One number per observation — risk, error, or outlier score | [→ page](glossary/score.md) |
| [`worse`](#worse) | Which direction is harmful (`higher` / `lower`) | [→ page](glossary/worse.md) |
| [Harmful shift](#harmful-shift) | Target excess mass in the harmful tail | [→ page](glossary/harmful-shift.md) |
| [Domain probability](#domain-probability) | `P(target|x)` for weighting | [→ page](glossary/domain-probability.md) |
| [Common support](#common-support) | Overlap region where weighting focuses the test | [→ page](glossary/common-support.md) |
| [Importance weights](#importance-weights) | `ImportanceWeights` normalized per group | [→ page](glossary/importance-weights.md) |
| [Reweight](#reweight) | Which group(s) to adjust (`source`/`target`/`both`) | [→ page](glossary/reweight.md) |
| [Shrinkage](#shrinkage) | `λ` blending plain ratio → uniform (default `0.5`) | [→ page](glossary/shrinkage.md) |
| [ESS](#effective-sample-size-ess) | Kish `(sum w)²/sum w²` — diagnostics | [→ page](glossary/ess.md) |
| [Permutation test](#permutation-test) | Labels permuted, scores fixed | [→ page](glossary/permutation-test.md) |
| [Honest scores](#honest-out-of-sample-scores) | Out-of-sample via CV / OOB / held-out | [→ page](glossary/honest-scores.md) |
| [Statistic vs p-value](#statistic-vs-p-value) | `.pvalue` first, `.statistic` second | [→ page](glossary/statistic-pvalue.md) |

## Source and target

**Source** — reference distribution (training or past batch). **Target** — evaluation distribution (production or new batch). Both are one numeric score per observation. Use `source=` / `target=` consistently; `group`/`membership_prob` are historic names.

→ [Details: Source and target](glossary/source-target.md)

## Score

A scalar signal per observation — predicted risk, prediction error, or outlier score (e.g., `LogitGap`). Any scalar works; `samesame` only tests it. Package term is *outlier score*.

→ [Details: Score](glossary/score.md)

## `worse`

Polarity that defines harmful direction for `test_harmful_shift` (string or `ss.Worse`, interchangeable). `higher` = larger is worse; `lower` = smaller is worse. Internally negated so larger always means worse.

→ [Details: `worse`](glossary/worse.md) · See [Shift testing](../api/testing.md).

## Harmful shift

Directional change where target has excess mass in the harmful tail you declared with `worse`. Distinct from *any* shift (`test_shift`, two-sided).

→ [Details: Harmful shift](glossary/harmful-shift.md)

## Domain probability

`P(target | x)` from a domain classifier. Passed as separate `source`/`target` arrays to `domain_weights(...)`; prior ratio inferred from lengths. For weighting, not a harm signal.

→ [Details: Domain probability](glossary/domain-probability.md)

## Common support

Overlap where both groups have density. Low-overlap points can dominate an unweighted test; weighting reframes to common support.

→ [Details: Common support](glossary/common-support.md)

## Importance weights

`ImportanceWeights` dataclass with `.source`/`.target`, normalized to group size. Built via `domain_weights(...)` or custom.

→ [Details: Importance weights](glossary/importance-weights.md) · [API](../api/weighting.md)

## Reweight

Policy `reweight="source"` / `"target"` / `"both"` (default) — which group(s) to adjust. Inactive groups get weight `1`.

→ [Details: Reweight](glossary/reweight.md)

## Shrinkage

`shrinkage` λ in `[0, 1]` (RIW) — `0` = plain ratio, `1` = uniform, `0.5` = default. Trades correction against stability.

→ [Details: Shrinkage](glossary/shrinkage.md)

## Effective sample size (ESS)

Kish’s ESS ` (sum w)² / sum w²` per group. `n` for uniform, `→1` when concentrated. Rule of thumb: worry when `ESS < n/4`.

--8<-- "snippets/ess-rule.txt"

→ [Details: ESS](glossary/ess.md) · [Diagnose](../examples/weighting/diagnose-weight-concentration.md)

## Permutation test

Label-permutation null; scores/weights fixed; `n_resamples` permutations. `test_shift` two-sided AUC, `test_harmful_shift` one-sided harm.

--8<-- "snippets/n-resamples.txt"

→ [Details: Permutation test](glossary/permutation-test.md)

## Honest (out-of-sample) scores

Must be `cross_val_predict` / `oob_decision_function_` / held-out. In-sample invalidates the test.

→ [Details: Honest scores](glossary/honest-scores.md)

## Statistic vs p-value

`.pvalue` (evidence, ≤0.05) first; `.statistic` (magnitude) second. AUC `0.5` = chance; harm has no fixed scale — compare to null median.

→ [Details: Statistic vs p-value](glossary/statistic-pvalue.md)

## References

- Yamada et al. (2013). Relative density-ratio estimation. *Neural Computation*.
- Shimodaira (2000). Improving predictive inference under covariate shift. *JSPI*.
- Kamulete et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
- Kish (1965). *Survey Sampling*. (ESS).
- Phipson & Smyth (2010). Permutation p-values.
