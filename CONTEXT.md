# Shift Testing & Context Weighting

This context defines the domain language for implementing paper-aligned weighting methods in samesame and translating them into product requirements.

## Language

**Subgroup Testing Module**:
A new top-level submodule (`samesame.subgroup`) that implements price-sensitivity heterogeneity testing for two-arm randomized pricing experiments, based on the Watson & Holmes (2020) statistical framework, taking `(y, treatment, X)` inputs where `y` is the binary purchase decision and `treatment` is the binary pricing arm assignment. Architecturally separate from the distribution-shift API.
_Avoid_: Extending test_shift, repurposing WeightingStrategy for this; applying to adaptive/bandit pricing logs without IPS correction

**Crossover price-sensitivity segment**:
A customer segment in a two-arm pricing experiment where the optimal pricing arm allocation differs; tested via a two-sample hypothesis test on held-out subgroup predictions. The pricing analogue of crossover TEH in Watson & Holmes (2020).
_Avoid_: Qualitative interaction (Gail-Simon framing)

**Non-crossover price-sensitivity**:
A type of price-sensitivity heterogeneity where one pricing arm is everywhere superior but the conversion lift varies systematically across customers; tested by stacking ML predictions against a baseline GLM. The pricing analogue of non-crossover TEH in Watson & Holmes (2020).
_Avoid_: Quantitative interaction

**Pricing Arm**:
The binary experimental assignment `treatment ∈ {0, 1}` indicating which price policy a customer was randomly allocated to in a two-arm pricing experiment. The module is policy-agnostic — arm semantics are the user's responsibility. Must be fully randomized: `P(treatment=1 | X) = 0.5` for all customers.
_Avoid_: Treatment group, ad arm, intervention arm

**Purchase Decision**:
The binary outcome variable `y ∈ {0, 1}` representing whether a customer completed a purchase (1) or not (0) in a pricing experiment. This is the only supported outcome type in v1 of `samesame.subgroup`.
_Avoid_: Revenue, conversion rate (use as aggregate statistic only), click

**Aggregate p-value**:
A single p-value combining split-wise evidence from K balanced two-fold data splits, computed as `min(1, Q_alpha({2*p_i}))` where Q_alpha is the alpha-quantile (default alpha=0.5, the median).
_Avoid_: Combined p-value, meta-analytic p-value
A release goal focused on adding new capability, not only stabilizing existing behavior.
_Avoid_: Hardening-only milestone, maintenance-only release

**Paper-Aligned Method**:
A method whose mathematical form and usage semantics are traceable to the target publication.
_Avoid_: Paper-inspired tweak, approximate variant

**Domain Probability**:
The per-sample probability that an observation belongs to the target group, constrained to the open interval (0, 1). Produced by a **Domain Classifier** and passed to `contextual_weights` as two separate arrays: `source_prob` (probabilities for source samples) and `target_prob` (probabilities for target samples). The prior ratio is always inferred from `len(source_prob) / len(target_prob)` — never supplied explicitly.
_Avoid_: Context Membership Probability (superseded), logit score, raw classifier margin, pooled flat array passed with a hidden ordering invariant

**Context-Aware Weighting Mode**:
A named policy (`'source'`, `'target'`, `'both'`) that controls which group's samples are reweighted by `contextual_weights`. Passed as the `mode` parameter.
_Avoid_: Ad hoc weighting, custom formula

## Relationships

- A **Feature Expansion Milestone** may include one or more **Paper-Aligned Methods**.
- A **Paper-Aligned Method** can require one or more **Context-Aware Weighting Modes**.
- A **Context-Aware Weighting Mode** consumes **Domain Probabilities**.

## Example dialogue

> **Dev:** "For this release, are we only cleaning docs and tests?"
> **Domain expert:** "No, this is a **Feature Expansion Milestone** and must deliver additional **Paper-Aligned Methods**."

## Flagged ambiguities

- "Implement the paper" could mean hardening existing code or adding new methods; resolved: this work is a **Feature Expansion Milestone**.
- Scope of "implement the paper" resolved: all three method components are in-scope: (1) crossover TEH test via repeated balanced two-fold data-splitting, (2) non-crossover TEH test via ML-stacking against a baseline model, (3) aggregate p-value construction from split-wise p-values for strict type I error control.
- "sample weight" was used loosely for both user-supplied weights and computed importance weights — resolved: `SampleWeighting` is the explicit user-supplied strategy; importance weights are always derived from domain probabilities via RIW.
- "statistic" appears both as the test statistic name (a string like `"roc_auc"`) and as the computed numeric value — context distinguishes them; `statistic_name` and `statistic` (float) are the canonical field names.
- "pricing experiment" could mean a fully randomized A/B test or an adaptive/contextual bandit; resolved: `samesame.subgroup` is only valid for **fully randomized two-arm experiments** where `P(treatment=1 | X) = 0.5`. Logs from adaptive or contextual pricing policies require IPS correction before use and are explicitly out of scope for v1.
- `alpha_blend` was the original parameter name for the RIW blending coefficient; resolved: renamed to `lambda_` (public-facing) to align with domain notation. `balance: bool` was a toggle for prior-ratio inference; resolved: always inferred from group sizes — removed entirely. `group`/`membership_prob` positional parameters for `contextual_weights` replaced by keyword-only `source_prob`/`target_prob` to make the source-first ordering invariant structural rather than documented.

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
The primary importance weighting strategy. Stabilises plain density-ratio weighting by blending source and target distributions in the denominator. Controlled by `lambda_` (public parameter name) / `lam` (internal variable name). Default `lambda_=0.5`.
_Avoid_: RIWERM (internal paper term, not user-facing), `alpha_blend` (superseded name)

**Weighting strategy**:
A tagged choice among: no weighting, explicit per-sample weights (`SampleWeighting`), or contextual RIW (`ContextualRIWWeighting`). Represented as a frozen dataclass union.
_Avoid_: weight mode, weighting method

**Domain classifier**:
A binary probabilistic classifier trained to distinguish source from target samples. Its out-of-bag or held-out predicted probabilities are the **Domain Probabilities** consumed by `contextual_weights`. Any calibrated binary classifier (e.g. random forest with OOB scores, logistic regression) may serve as the domain classifier; `samesame` is agnostic to the choice.
_Avoid_: membership classifier (superseded), two-sample discriminator
