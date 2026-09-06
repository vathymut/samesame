# Core concepts

A score, a split, and a declaration: the three choices that shape every `samesame` comparison.

## Context: why these concepts matter

Production monitoring is constrained: features are wide, labels arrive late, and deployment populations shift. A single **outlier score** `ϕ(x)` per observation reduces each row to one number you can test (predicted risk, error, or confidence gap). Every conclusion is relative: **source** (reference) versus **target** (current deployment). Whether a shift is harmful depends on which tail you declare.

## What

--8<-- "snippets/source-target.txt"

**Score `ϕ(x)` and outlier score**: one interpretable scalar per observation. Choose `ϕ` to encode the outcome you care about. Outlier scores (confidence, typicality) follow the same rule: larger `ϕ` means more distant from source.

**Worse**: the polarity that defines harm. Declare it from the meaning of the score *before* you look. Pass as `worse="higher"` / `"lower"` or `ss.Worse.HIGHER` / `LOWER` — interchangeable:

--8<-- "snippets/worse-table.txt"

**Domain probability `P(target|x)`**: probability from a domain classifier that an observation belongs to target. It is a membership score, not outcome quality — keep it separate from the `ϕ` you test.

**Reweight and common support**: when groups barely overlap, a few points dominate. Weighting reframes the comparison around shared regions; it creates no information and changes the population you describe. Start unweighted.

--8<-- "snippets/reweight-table.txt"

**Shrinkage `λ`**: bias–variance trade in density-ratio estimation. `λ=0` strongest correction, `λ=1` uniform.

--8<-- "snippets/shrinkage-table.txt"

**Effective sample size (ESS)**: `(Σw)²/Σw²` (Kish 1965). Uniform weights give `ESS=n`; concentration drives it toward 1. `ESS/n` well below `0.5` warns the result rests on few points. If low even at `λ=0.5`, keep the comparison unweighted.

**Honest scores**: valid p-values need out-of-sample scores. --8<-- "snippets/honest-scores.txt"

**Reading p-values**: --8<-- "snippets/pvalue-caveat.txt"

## How it fits

- *Did it change?* `ss.test_shift` asks any shift (ROC AUC `0.5` = no separation, two-sided).
- *Did it get worse?* `ss.test_harmful_shift(..., worse=...)` asks tail harm (weighted AUC, one-sided `greater`).
- *Is the comparison fair?* `ss.domain_weights` + `ESS` asks whether the shift survives on common support.

## Related

- [Get started](../examples/tutorials/get-started.md): run both tests hands-on.
- [Weight for common support](../how-to/weight-for-common-support.md): when and how to weight.
- [How the harm test works](harmful-shift-statistic.md): why the weighted AUC emphasizes the harmful tail.
- [Shift testing](../api/testing.md) · [Importance weights](../api/weighting.md): reference.
