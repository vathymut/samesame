# Shift Testing & Context Weighting

This context defines the domain language for implementing paper-aligned weighting methods in samesame and translating them into product requirements.

## Language

**TEH Module**:
A new top-level submodule (`samesame.teh`) that implements treatment effect heterogeneity testing from Watson & Holmes (2020), taking `(y, treatment, X)` inputs. Architecturally separate from the distribution-shift API.
_Avoid_: Extending test_shift, repurposing WeightingStrategy for this
A release goal focused on adding new capability, not only stabilizing existing behavior.
_Avoid_: Hardening-only milestone, maintenance-only release

**Paper-Aligned Method**:
A method whose mathematical form and usage semantics are traceable to the target publication.
_Avoid_: Paper-inspired tweak, approximate variant

**Context Membership Probability**:
The per-sample probability that an observation belongs to the target group, constrained to the open interval (0, 1).
_Avoid_: Logit score, raw classifier margin

**Context-Aware Weighting Mode**:
A named policy that maps membership probabilities and group labels into sample weights for shift testing.
_Avoid_: Ad hoc weighting, custom formula

## Relationships

- A **Feature Expansion Milestone** may include one or more **Paper-Aligned Methods**.
- A **Paper-Aligned Method** can require one or more **Context-Aware Weighting Modes**.
- A **Context-Aware Weighting Mode** consumes **Context Membership Probabilities**.

## Example dialogue

> **Dev:** "For this release, are we only cleaning docs and tests?"
> **Domain expert:** "No, this is a **Feature Expansion Milestone** and must deliver additional **Paper-Aligned Methods**."

## Flagged ambiguities

- "Implement the paper" could mean hardening existing code or adding new methods; resolved: this work is a **Feature Expansion Milestone**.
- Scope of "implement the paper" resolved: all three method components are in-scope: (1) crossover TEH test via repeated balanced two-fold data-splitting, (2) non-crossover TEH test via ML-stacking against a baseline model, (3) aggregate p-value construction from split-wise p-values for strict type I error control.
- "sample weight" was used loosely for both user-supplied weights and computed importance weights — resolved: `SampleWeighting` is the explicit user-supplied strategy; importance weights are always derived from membership probabilities via RIW.
- "statistic" appears both as the test statistic name (a string like `"roc_auc"`) and as the computed numeric value — context distinguishes them; `statistic_name` and `statistic` (float) are the canonical field names.

## Core API language (distribution shift)

**Outlier score**:
A scalar signal from a model indicating how anomalous an input is.
_Avoid_: anomaly score (ambiguous), OOD score (too specific)

**Source**:
The baseline distribution of outlier scores, typically from training or reference data.
_Avoid_: reference distribution, in-distribution

**Target**:
The new distribution of outlier scores compared against source, typically from production or test data.
_Avoid_: test set, deployment data

**Shift**:
Any detectable difference between source and target score distributions.
_Avoid_: drift (implies temporal), covariate shift (implies specific mechanism)

**Adverse shift**:
A shift in the harmful direction — scores moving toward higher risk or lower confidence. Requires a declared direction.
_Avoid_: bad shift, harmful drift

**Direction**:
Whether higher outlier scores indicate worse outcomes (`higher-is-worse`) or better outcomes (`higher-is-better`). Required for adverse shift testing.
_Avoid_: polarity, orientation

**Importance weight**:
A per-sample weight used to correct for covariate shift between source and target during a shift test.
_Avoid_: reweighting factor

**RIW (Relative Importance Weight)**:
The primary importance weighting strategy. Stabilises plain density-ratio weighting by blending source and target distributions in the denominator. Controlled by `lam`.
_Avoid_: RIWERM (internal paper term, not user-facing)

**Weighting strategy**:
A tagged choice among: no weighting, explicit per-sample weights (`SampleWeighting`), or contextual RIW (`ContextualRIWWeighting`). Represented as a frozen dataclass union.
_Avoid_: weight mode, weighting method
