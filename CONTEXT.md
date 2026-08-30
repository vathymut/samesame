# samesame Package Context

This context records stable domain language for `samesame`. Use it when
choosing terms in code, tests, docs, or examples.

## Scope

`samesame` ships score-based source-versus-target monitoring for detecting
distribution changes in ML deployments.

**In scope:**
- Detecting harmful shift in scores derived from trained models (e.g., predicted
  risk, confidence, error rates)
- Reweighting comparisons to focus on regions of common support between source
  and target distributions
- Permutation-based inference with importance weights

**Out of scope:**
- Randomized-experiment validation
- Subgroup discovery or model-lift comparison
- Sequential monitoring (time-series alarms)

## Audience

Primary audience: **ML engineers and data scientists with statistical literacy**.

Assume they understand:
- p-values and hypothesis testing
- Supervised learning workflows (training/deployment, predictions, scores)
- Distribution comparison concepts (shift, covariate shift, label shift)

## Terminology

### Core workflow terms

**Source** and **Target**: The two groups being compared. Source is the
reference distribution (typically training or past deployment); target is the
distribution under evaluation (typically current deployment).

**Harmful shift**: A directional distributional change where the target group
shows movement toward a worse outcome, governed by the polarity of the score.

**worse**: The polarity parameter that defines harmful movement. Use
`worse="higher"` when larger scores indicate harm (e.g., predicted risk).
Use `worse="lower"` when smaller scores indicate harm (e.g., lower confidence).

### Weighting terms

**Domain probability**: The probability, output by a domain classifier, that an
observation belongs to the target group.

**Reweight**: The weighting policy that controls which group(s) are adjusted
toward common support:
- `"source"`: reweight source samples to match target
- `"target"`: reweight target samples to match source
- `"both"`: reweight both groups (common-support comparison)

**Shrinkage**: The shrinkage parameter `shrinkage` (λ) in [0, 1] that controls the
bias-variance tradeoff in RIW (Relative Importance Weight) estimation.
`shrinkage=0` gives plain density-ratio weights; `shrinkage=1` gives uniform
weights; `shrinkage=0.5` is the recommended default.

**ImportanceWeights**: Per-group importance weights quantifying common support,
normalized so each group's weights sum to its sample size.

### Example-specific terms

**Outlier score**: The package's preferred term for a scalar anomaly-like signal
from a model. Used in examples (e.g., confidence monitoring). Avoid "anomaly
score" or "OOD score" in package documentation.
