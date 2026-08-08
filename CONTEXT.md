# samesame Package Context

This context records the package-facing language and standing decisions for
`samesame`.

Use it when editing code, tests, docs, or examples under the supported package
surface: `samesame.shift` and `samesame.weights`.

For manuscript work under `research/papers/dw/`, use
`research/papers/dw/CONTEXT.md`.

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

Keep out-of-scope work in sibling packages. Do not add `subgroup`,
`model_selection`, or `sequential` namespaces to `samesame`.

## Audience

Primary audience: **ML engineers and data scientists with statistical literacy**.

Assume they understand:
- p-values and hypothesis testing
- Supervised learning workflows (training/deployment, predictions, scores)
- Distribution comparison concepts (shift, covariate shift, label shift)

Reach business stakeholders through worked examples, not by flattening the main
documentation.

## Terminology

### Core workflow terms

**Source** and **Target**: The two groups being compared. Source is the
reference distribution (typically training or past deployment); target is the
distribution under evaluation (typically current deployment).

**Harmful shift**: A directional distributional change where the target group
shows movement toward a worse outcome. Detected via
`shift.detect_harm(source_scores, target_scores, direction=...)`.

**Direction**: The polarity parameter that defines "worse." Use
`"higher-is-worse"` when larger scores indicate harm (e.g., predicted risk).
Use `"higher-is-better"` when larger scores indicate quality (e.g., confidence,
accuracy).

### Weighting terms

**Domain probability**: The probability, output by a domain classifier, that an
observation belongs to the target group. Passed as separate `source_prob` and
`target_prob` arrays to `from_domain_probabilities(...)`. The prior ratio is
always inferred from group sizes.

**Mode**: The weighting policy passed to `from_domain_probabilities(...)`:
- `"source"`: reweight source samples to match target
- `"target"`: reweight target samples to match source
- `"both"`: reweight both groups (common-support comparison)

**lambda_**: The stabilization parameter in [0, 1] that controls the
bias-variance tradeoff in RIW (Relative Importance Weight) estimation.
`lambda_=0` gives plain density-ratio weights; `lambda_=1` gives uniform
weights; `lambda_=0.5` is the recommended default.

**ImportanceWeights**: A frozen dataclass holding `.source` and `.target` weight
arrays, typically built via `from_domain_probabilities(...)`. Weights are
normalized so each group sums to its sample size.

### Example-specific terms

**Outlier score**: The package's preferred term for a scalar anomaly-like signal
from a model. Used in examples (e.g., confidence monitoring). Avoid "anomaly
score" or "OOD score" in package documentation.

## API conventions

- **Keyword-only parameters**: All `from_domain_probabilities(...)` parameters
  are keyword-only (no positional args after `*`).
- **Return types**: Results are frozen dataclasses (`ShiftResult`, `HarmResult`,
  `ImportanceWeights`), not dicts or tuples.
- **Naming pattern**: Test functions use `statistic` for the numeric value and
  `statistic_name` for the string identifier.
- **Historical renames**: `alpha_blend` → `lambda_`; `balance` (removed,
  inferred); `group`/`membership_prob` → `source_prob`/`target_prob`.

## Non-package guidance

**Example-specific decisions** (e.g., `LogitGap` recipe, confidence monitoring
workflow) live in `docs/how-to/` frontmatter or inline comments, not here.

**Research code conventions** (e.g., `typer` CLIs, `polars` usage, `_dgp.py`
module structure) belong in `research/papers/dw/CONTEXT.md`, not here.
