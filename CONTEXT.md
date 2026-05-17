# Shift Testing & Context Weighting

This context defines the domain language for implementing paper-aligned weighting methods in samesame and translating them into product requirements.

## Language

**RCT Validation Modules**:
Two new top-level submodules for validating two-arm randomised experiments, based on the Watson & Holmes (2020) statistical framework: `samesame.subgroup` (`test_subgroup_effect`) tests for crossover heterogeneity; `samesame.model_selection` (`test_model_lift`) tests whether an estimator captures more heterogeneity than a reference estimator. Both take `(y, treatment, features)` inputs where `y` is the binary outcome and `treatment` is the binary arm assignment. Domain-agnostic and architecturally separate from the distribution-shift API.
_Avoid_: Collapsing both into a single `samesame.subgroup` module; extending `shift.detect_shift`; repurposing WeightingStrategy for this; applying to adaptive/bandit logs without IPS correction; hardcoding pricing vocabulary in public parameter names

**Crossover Subgroup**:
A subgroup in a two-arm experiment where the optimal arm allocation differs across members; the effect of `test_subgroup_effect()`. The general analogue of crossover TEH in Watson & Holmes (2020) — that paper term is acceptable in docstrings and internal comments as a reference, but must not appear in public-facing API names or user documentation.
_Avoid_: Qualitative interaction (Gail-Simon framing), price-sensitive segment (domain-specific)

**Non-crossover Heterogeneity**:
A type of treatment effect heterogeneity where one arm is everywhere superior but the *degree* of benefit varies systematically across subjects. The effect detected by `samesame.model_selection.test_model_lift()`: it tests whether the estimator predicts *who benefits more from the superior arm* better than a reference estimator, not whether it predicts the outcome generally. The general analogue of non-crossover TEH in Watson & Holmes (2020) — that paper term is acceptable in docstrings and internal comments as a reference, but must not appear in public-facing names or user documentation.
_Avoid_: "captures more heterogeneity" (ambiguous — implies general predictive quality, not treatment-benefit prediction); quantitative interaction; non-crossover price-sensitivity (domain-specific)

**Public API: samesame.subgroup**:
`samesame.subgroup.test_subgroup_effect(y, treatment, features, estimator, ...)` — tests for crossover heterogeneity (real subgroups where arms differ in direction). Returns a `SubgroupResult` with `.pvalue`. The `test_*` prefix is consistent with `shift.detect_shift` / `shift.detect_harm` in the parent package.
_Avoid_: `test_segments`, `test_heterogeneity` (superseded); `ml_model` as parameter name (superseded by `estimator`); surface paper term `crossover TEH` only in docstring `Notes` sections

**Public API: samesame.model_selection**:
`samesame.model_selection.test_model_lift(y, treatment, features, estimator, reference_estimator, ...)` — tests whether the estimator predicts who benefits from the superior arm better than the reference estimator (non-crossover heterogeneity). Returns a `SubgroupResult` with `.pvalue`. Future additions to `samesame.model_selection` may include calibration comparison and AUC-based lift tests.
_Avoid_: "captures more heterogeneity" (inaccurate framing — the test is specifically about treatment-benefit prediction, not general predictive quality); `test_model_improvement`, `test_heterogeneity` (superseded); `ml_model` / `reference_model` as parameter names (superseded by `estimator` / `reference_estimator`); surface paper term `non-crossover TEH` only in docstring `Notes` sections

**Test ordering**:
`samesame.model_selection.test_model_lift()` and `samesame.subgroup.test_subgroup_effect()` are independent tools. No sequential ordering is prescribed. Use `test_model_lift` to validate that an estimator captures treatment benefit better than a reference estimator (non-crossover heterogeneity). Use `test_subgroup_effect` to determine whether genuine crossover subgroups exist (crossover heterogeneity). Either may be used in isolation. If both are used, the order depends on the practitioner's question, not on a methodological requirement from the paper.
_Avoid_: Prescribing a fixed order in docs; presenting them as a mandatory two-step workflow

**Unfitted Estimator**:
The `estimator` and `reference_estimator` parameters of `test_subgroup_effect()` and `test_model_lift()` accept **unfitted** sklearn-compatible estimators (e.g., `RandomForestClassifier(n_estimators=100)`, `LogisticRegression()`). Any supervised classifier with `.predict_proba()` is valid — statistical models and ML models alike. The function owns all fitting internally across K splits using `sklearn.base.clone()` to prevent data leakage. Passing a pre-fitted model would leak the training data into the held-out test splits and invalidate the p-value.
_Avoid_: `ml_model` / `reference_model` as parameter names (superseded); fitted estimator as input; "trained model" language in user-facing docs; do not use `clone()` only on some splits

**RCT Validation modules docs placement**:
Each RCT validation module gets its own independent Tutorial page and API reference page. `samesame.model_selection` (v1.0): Tutorial *"Validate a pricing experiment"* and API reference `api/model_selection.md`. `samesame.subgroup` (v1.1): Tutorial *"Detect subgroup treatment effects"* and API reference `api/subgroup.md`. No shared Tutorial. No How-to guide in v1. Both API pages mirror `api/testing.md` in structure. The `site_description` in `mkdocs.yml` must be updated to cover RCT validation alongside distribution shift.
_Avoid_: A shared Tutorial page covering both modules as a sequential workflow; adding either module to How-to guides only; skipping the Tutorial; leaving the site_description shift-only

**RCT Validation modules experimental status**:
Both `samesame.subgroup` and `samesame.model_selection` are marked "experimental" via a note in each module-level docstring and in their respective API reference pages — not via `warnings.warn` at import time. A runtime warning on every import is noisy and gets suppressed immediately. "Experimental" here means "API may change before v2.0 stability guarantee," not "may corrupt data." Revisit at v1.1.
_Avoid_: `warnings.warn` at import; silently omitting the experimental caveat from docs; marking only one of the two modules experimental

**RCT Validation module namespaces**:
`samesame.subgroup` and `samesame.model_selection` are each exposed via `from . import subgroup` and `from . import model_selection` in `__init__.py`. Functions are **not** hoisted to `samesame.*`. Users call `samesame.subgroup.test_subgroup_effect(...)` and `samesame.model_selection.test_model_lift(...)` — consistent with how `samesame.weights` is exposed. Both modules solve a structurally different problem (RCT validation) from the shift module (distribution monitoring); keeping them as separate namespaces makes the functional distinction explicit and lets both stay "experimental" without contaminating the stable top-level API.
_Avoid_: Hoisting `test_subgroup_effect` or `test_model_lift` into the `samesame.*` top-level namespace; merging both into a single `samesame.subgroup` namespace

**Null calibration test threshold**:
The null calibration test in `tests/test_subgroup.py` uses 200 null datasets and a threshold of ≤ 10% false positives (p < 0.05). At n=200, a 10% bound corresponds to p < 0.001 under the null binomial — making a false alarm extremely unlikely if the implementation is correctly calibrated. The PRD's AC-05 value of "≤ 5%" is incorrect: it would cause the test to fail ~50% of the time on a correctly calibrated implementation by pure chance. Do not re-tighten to 5%.
_Avoid_: 100 null datasets with ≤ 5% threshold (AC-05 value, superseded); setting threshold equal to the nominal alpha

**RCT Validation modules parameters**:
The `n_splits` (default 200) and `random_state` (default `None`) parameters follow sklearn conventions throughout both `samesame.subgroup` and `samesame.model_selection`. `random_state` accepts `int | np.random.RandomState | None` — identical to sklearn's contract. Using sklearn conventions prevents silent `TypeError`s for users who already know the sklearn vocabulary.
_Avoid_: `num_splits` (PRD name, superseded), `random_seed` (PRD name, superseded); accepting only `int` for `random_state`

**RCT Validation modules file layout**:
Each module lives in its own single file: `src/samesame/subgroup.py` (contains `test_subgroup_effect` and its private helpers) and `src/samesame/model_selection.py` (contains `test_model_lift` and its private helpers). No separate internals file for either. Both modules may import from `samesame._utils` (e.g., `as_numeric_vector`, binary validation via `type_of_target`) and may reuse public result types already owned by public seams (for example `TestResult` from `samesame.shift`). If a shared `SubgroupResult` becomes necessary once both modules exist, extract it then rather than relying on a pre-emptive `_types` module. "Standalone" means no dependency on distribution-shift logic in `shift.py`, on weighting strategies (`weights.py`), or on each other.
_Avoid_: Putting both public functions in a single `subgroup.py`; creating a separate `_subgroup_internals.py`; importing from `_api.py`, `_data.py`, `_metrics.py`, or `weights.py`; duplicating binary-validation logic already in `_utils.py`; cross-importing between `subgroup.py` and `model_selection.py`

**Treatment Arm**:
The binary experimental assignment `treatment ∈ {0, 1}` indicating which arm a subject was randomly allocated to in a two-arm experiment. The parameter is named `treatment` in the public API. The module is domain-agnostic — arm semantics are the user's responsibility. Must be fully randomised: `P(treatment=1 | X) = 0.5` for all subjects.
_Avoid_: `pricing_arm` (domain-specific parameter name), treatment group, intervention arm

**Binary Outcome**:
The outcome variable `y \u2208 {0, 1}` passed as the first argument to `samesame.subgroup` and `samesame.model_selection` functions. Domain-agnostic: could represent a purchase, default, conversion, or any binary event. This is the only supported outcome type in v1. In the pricing motivating example this is a purchase decision.
_Avoid_: `purchase_decision` (domain-specific parameter name), revenue, conversion rate (use as aggregate statistic only)

**Aggregate p-value**:
A single p-value combining split-wise evidence from K balanced two-fold data splits, computed as `min(1, Q_alpha({2*p_i}))` where Q_alpha is the alpha-quantile (default alpha=0.5, the median).
_Avoid_: Combined p-value, meta-analytic p-value
A release goal focused on adding new capability, not only stabilizing existing behavior.
_Avoid_: Hardening-only milestone, maintenance-only release

**Paper-Aligned Method**:
A method whose mathematical form and usage semantics are traceable to the target publication.
_Avoid_: Paper-inspired tweak, approximate variant

**Domain Probability**:
The probability for each observation that it belongs to the target group, constrained to the open interval (0, 1). Produced by a **Domain Classifier** and passed to `from_domain_probabilities` as two separate arrays: `source_prob` (probabilities for source samples) and `target_prob` (probabilities for target samples). The prior ratio is always inferred from `len(source_prob) / len(target_prob)` — never supplied explicitly.
_Avoid_: Context Membership Probability (superseded), logit score, raw classifier margin, pooled flat array passed with a hidden ordering invariant

**Context-Aware Weighting Mode**:
A named policy (`'source'`, `'target'`, `'both'`) that controls which group's samples are reweighted by `from_domain_probabilities`. Passed as the `mode` parameter.
_Avoid_: Ad hoc weighting, custom formula

**Primary Audience**:
The primary audience of `samesame` is **ML engineers and data scientists with a statistical background** — people who understand p-values, train supervised models, and run A/B experiments. Documentation and API design should respect this literacy: no need to explain what a p-value is, what a supervised model is, or what a two-arm experiment is. Business stakeholders (product managers, executives) are a **secondary audience** reached indirectly via the worked examples that practitioners share with them — not via dumbing down the primary docs.
_Avoid_: Spoon-feeding p-value definitions in API docstrings or primary docs; writing tutorials at a level that assumes no statistical background

**SubgroupResult**:
The return type of `test_subgroup_effect()` and `test_model_lift()`. Extends `TestResult` (inherits `.statistic: float` and `.pvalue: float`) and adds `.null_distribution: NDArray[np.float64]`. The `.statistic` field holds the aggregate test statistic value (the combined evidence before p-value conversion). Consistent with `ShiftResult` and `HarmResult` in shape.
_Avoid_: Standalone result type with no `TestResult` inheritance; naming it `Result` (too generic)

## Relationships

- A **Feature Expansion Milestone** may include one or more **Paper-Aligned Methods**.
- A **Paper-Aligned Method** can require one or more **Context-Aware Weighting Modes**.
- A **Context-Aware Weighting Mode** consumes **Domain Probabilities**.
- `samesame.subgroup.test_subgroup_effect()` and `samesame.model_selection.test_model_lift()` both return a **SubgroupResult**, which extends **TestResult**.

## Example dialogue

> **Dev:** "For this release, are we only cleaning docs and tests?"
> **Domain expert:** "No, this is a **Feature Expansion Milestone** and must deliver additional **Paper-Aligned Methods**."

## Flagged ambiguities

- "Implement the paper" could mean hardening existing code or adding new methods; resolved: this work is a **Feature Expansion Milestone**.
- Scope of "implement the paper" resolved: all three method components are in-scope: (1) crossover TEH test via repeated balanced two-fold data-splitting, (2) non-crossover TEH test via ML-stacking against a baseline model, (3) aggregate p-value construction from split-wise p-values for strict type I error control.
- "sample weight" was used loosely for both user-supplied weights and computed importance weights — resolved: `SampleWeighting` is the explicit user-supplied strategy; importance weights are always derived from domain probabilities via RIW.
- "statistic" appears both as the test statistic name (a string like `"roc_auc"`) and as the computed numeric value — context distinguishes them; `statistic_name` and `statistic` (float) are the canonical field names.
- "pricing experiment" could mean a fully randomized A/B test or an adaptive/contextual bandit; resolved: `samesame.subgroup` is only valid for **fully randomized two-arm experiments** where `P(treatment=1 | X) = 0.5`. Logs from adaptive or contextual pricing policies require IPS correction before use and are explicitly out of scope for v1.
- `alpha_blend` was the original parameter name for the RIW blending coefficient; resolved: renamed to `lambda_` (public-facing) to align with domain notation. `balance: bool` was a toggle for prior-ratio inference; resolved: always inferred from group sizes — removed entirely. `group`/`membership_prob` positional parameters for `from_domain_probabilities` replaced by keyword-only `source_prob`/`target_prob` to make the source-first ordering invariant structural rather than documented.
- "change" is acceptable explanatory language for the generic **Shift** question in docs and tutorials (for example, "did anything change?"). Public API names remain `shift.detect_shift(...)` and `ShiftResult`, not `detect_change(...)` or `ChangeResult`.

## Core API language (distribution shift)

**Outlier score**:
A scalar signal from a model indicating how anomalous an input is.
_Avoid_: anomaly score (ambiguous), OOD score (too specific)

**Logit-derived Outlier score**:
An **Outlier score** computed directly from classifier logits; the only public score type in the narrowed score module. `LogitGap` and `MaxLogit` are the canonical examples.
_Avoid_: generic score array, confidence score (too vague)

**Source**:
The baseline distribution of outlier scores, typically from training or reference data.
_Avoid_: reference distribution, in-distribution

**Target**:
The new distribution of outlier scores compared against source, typically from production or test data.
_Avoid_: test set, deployment data

**Shift**:
Any detectable difference between source and target score distributions.
_Avoid_: drift (implies temporal), covariate shift (implies specific mechanism)

**Harmful shift**:
A shift in the harmful direction — scores moving toward higher risk or lower confidence. Requires a declared direction.
_Avoid_: adverse shift, bad shift, harmful drift

**Direction**:
Whether higher outlier scores indicate worse outcomes (`higher-is-worse`) or better outcomes (`higher-is-better`). Required for harmful shift testing.
_Avoid_: polarity, orientation

**Importance weight**:
A sample weight used to correct for covariate shift between source and target during a shift test.
_Avoid_: reweighting factor

**RIW (Relative Importance Weight)**:
The primary importance weighting strategy. Stabilises plain density-ratio weighting by blending source and target distributions in the denominator. Controlled by `lambda_` (public parameter name) / `lam` (internal variable name). Default `lambda_=0.5`.
_Avoid_: RIWERM (internal paper term, not user-facing), `alpha_blend` (superseded name)

**Weighting strategy**:
A tagged choice among: no weighting, explicit sample weights (`SampleWeighting`), or contextual RIW (`ContextualRIWWeighting`). Represented as a frozen dataclass union.
_Avoid_: weight mode, weighting method

**Domain classifier**:
A binary probabilistic classifier trained to distinguish source from target samples. Its out-of-bag or held-out predicted probabilities are the **Domain Probabilities** consumed by `from_domain_probabilities`. Any calibrated binary classifier (e.g. random forest with OOB scores, logistic regression) may serve as the domain classifier; `samesame` is agnostic to the choice.
_Avoid_: membership classifier (superseded), two-sample discriminator
