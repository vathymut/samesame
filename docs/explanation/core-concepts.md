# Core concepts

A score, a split, and a declaration: the three choices that shape every `samesame` comparison.

## Context: why these concepts matter

Production monitoring rarely gives you clean labels on time. Features are wide, deployment populations drift, and you still have to decide whether to act. A single **outlier score** `ϕ(x)` per observation reduces each row to one number you can test, whether that's predicted risk, error, or a confidence gap. Every conclusion is relative: **source** (your reference) versus **target** (current deployment). And whether a shift counts as harmful depends on which tail you name in advance.

## What

--8<-- "snippets/source-target.txt"

**Score `ϕ(x)` and outlier score**: one interpretable number per observation. Choose `ϕ` to encode the outcome you care about. Outlier scores for confidence or typicality follow the same rule: larger `ϕ` means further from source.

**Worse**: the polarity that defines harm. Declare it from what the score means *before* you look. Pass `worse="higher"` / `"lower"` or `ss.Worse.HIGHER` / `LOWER`; the two forms agree:

--8<-- "snippets/worse-table.txt"

**Domain probability `P(target|x)`**: the output of a domain classifier, estimating how likely an observation is to belong to target. It measures membership, not outcome quality, so keep it separate from the `ϕ` you test.

**Reweight and common support**: when the groups barely overlap, a few points can dominate. Weighting reframes the comparison around the ground they share and adds no information elsewhere. It changes the population you describe, so start unweighted.

--8<-- "snippets/reweight-table.txt"

**Shrinkage `λ`**: the bias-variance trade in density-ratio estimation. `λ=0` corrects hardest, `λ=1` stays uniform.

--8<-- "snippets/shrinkage-table.txt"

**Effective sample size (ESS)**: `(Σw)²/Σw²` (Kish 1965). Uniform weights give `ESS=n`; concentrated weights pull it toward 1. `ESS/n` well below `0.5` warns your result rests on a few points. If it stays low even at `λ=0.5`, keep the comparison unweighted.

**Honest scores**: valid p-values need out-of-sample scores. --8<-- "snippets/honest-scores.txt"

**Reading p-values**: --8<-- "snippets/pvalue-caveat.txt"

## How it fits

- *Did it change?* `ss.test_shift` looks for any shift (ROC AUC, where `0.5` means no separation, two-sided).
- *Did it get worse?* `ss.test_harmful_shift(..., worse=...)` looks for tail harm (weighted AUC, one-sided `greater`).
- *Is the comparison fair?* `ss.domain_weights` plus `ESS` asks whether the shift survives on common support.

## Related

- [Get started](../examples/tutorials/get-started.md): run both tests hands-on.
- [Weight for common support](../how-to/weight-for-common-support.md): when and how to weight.
- [How the harm test works](harmful-shift-statistic.md): why the weighted AUC emphasizes the harmful tail.
- [Shift testing](../api/testing.md) · [Importance weights](../api/weighting.md): reference.
