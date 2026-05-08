---
title: "PRD: Subgroup Testing Module — samesame.subgroup"
version: 1.1
date_created: 2026-05-01
date_revised: 2026-05-07
owner: Research Team
reference: "Watson & Holmes (2020): Machine learning analysis plans for randomised controlled trials"
tags: [prd, subgroup, rct, ab-test, feature-expansion, statistical-validation]
---

# PRD: Subgroup Testing Module

## 1. Executive Summary

### The Problem

**ML teams running A/B experiments have no way to validate whether their model found real heterogeneous treatment effects or statistical noise.**

When teams run two-arm experiments, data scientists often build ML models to find subgroups that respond differently to the two conditions. But without formal statistical validation, there is no way to know if the patterns are *real* or just **noise**. The consequences:

- Deploying strategies that only *appeared* to work in the experiment data (false positives)
- Missing real, profitable opportunities because we conflate luck with signal
- Undermining trust in the experimentation process across the organisation

**We are acting on ML subgroup findings with no statistical validation.** A model that identifies responsive subgroups might look compelling on the experiment data — but we cannot know whether it reflects a genuine effect or overfitting without a formal test. This is a material risk.

> **Motivating example**: A pricing team builds a Random Forest to find customers who respond differently to two price points. The model looks good. But is that response real, or did the model find a spurious pattern? This PRD addresses exactly this question — and generalises to any two-arm experiment: credit offers, product recommendations, clinical interventions, or any binary-outcome RCT.

### Proposed Solution

Implement a `samesame.subgroup` module — part of samesame's broader mission of *rigorous statistical comparison of groups for ML engineers and analysts* — that answers two concrete questions:

1. **Are there real subgroups with different responses to the two arms?** (Not just random noise)
2. **Does an ML model capture more of that heterogeneity than a reference model?** (With statistical proof)

The module uses a peer-reviewed statistical framework (Watson & Holmes 2020) that controls false positives by design. The API is simple: provide experiment data and an unfitted model spec; receive a p-value.

### Success Criteria

| # | Goal | How We Measure | Target |
|---|------|----------------|--------|
| SC-01 | Prevent false positives | Null calibration under controlled conditions | False positive rate ≤ 5% (tested at ≤ 10% with n=200 null datasets — see AC-05) |
| SC-02 | Catch real signals | Detect planted subgroup effects | ≥ 80% power at n=400 |
| SC-03 | No regressions | Existing samesame tests still pass | 100% pass rate |
| SC-04 | Reproducible results | Same data + same `random_state` = same result | Bit-for-bit identical |
| SC-05 | Fast enough | Large experiments don't time out | ≤ 60 seconds for n=500, `n_splits=200` |
| SC-06 | API simplicity | Easy for practitioners to use | Keyword-only parameters; unfitted estimator |

---

## 2. User Experience & Functionality

### User Personas

- **Business Stakeholder**: "I need to know if this experiment is safe to act on. Can we trust the conclusions?"
- **Product Manager**: "I need to know if the data science team's claim is valid before we allocate engineering resources."
- **Data Scientist / Analyst**: "I built a model that finds responsive subgroups. How do I prove it's not just overfitting?"

---

### The Core Questions We Answer

This module answers exactly **two** questions:

#### **Question 1: Are there real subgroups with different treatment responses?**
*Or did we just get lucky with noise?*

You give the module:
- Experiment data (subject features, treatment arm, binary outcome)
- An unfitted ML model spec (Random Forest, XGBoost, whatever)

The module returns:
- A p-value: "This result would happen by random chance ~X% of the time"
- If p < 0.05: You have evidence of real subgroups. Act with confidence.
- If p ≥ 0.05: This looks like noise. Don't act yet.

---

#### **Question 2: Does the ML model capture more heterogeneity than a reference model?**
*Not just "this model fits the data," but "this model detects more signal than a simple baseline"?*

You give the module:
- Experiment data
- An unfitted ML model spec
- An unfitted reference model spec (e.g., logistic regression)

The module returns:
- A p-value: "This difference would happen by random chance ~X% of the time"
- If p < 0.05: Your ML model detects more heterogeneity than the reference. It earns its complexity.
- If p ≥ 0.05: You don't have evidence yet. Collect more data or try a different model.

---

### User Stories

#### **Story 1: Test for real subgroup effects**

> As a data scientist, I want to validate that my ML model found actual subgroups with different treatment responses — not just patterns from random noise — so that I can confidently recommend a subgroup-based strategy to leadership.

**Acceptance Criteria:**

- `samesame.subgroup.test_subgroup_effect(y, treatment, features, ml_model)` runs without error
- Returns a `SubgroupResult` with `.pvalue`
- If I pass null (random) data, the p-value is ≥ 0.05 most of the time (prevents false alarms)
- If I pass data with a real crossover subgroup, the p-value is < 0.05 (catches real signals)
- Same input + same `random_state` always gives the same p-value (reproducible)
- Raises a clear `ValueError` if `treatment` is not binary (0 or 1)
- Raises a clear `ValueError` if `y` is not binary (0 or 1)
- Takes < 60 seconds for a 500-subject experiment with `n_splits=200`
- Works with any sklearn-compatible unfitted estimator

---

#### **Story 2: Validate an ML model against a reference model**

> As a product manager, I want a statistical test that proves our ML model would actually detect more heterogeneity than a simple reference model in a fresh experiment — not just that it looks better on the data we used — so I can confidently approve the engineering effort to deploy it.

**Acceptance Criteria:**

- `samesame.subgroup.test_model_lift(y, treatment, features, ml_model, reference_model)` runs without error
- Returns a `SubgroupResult` with `.pvalue`
- If both models are equivalent, the p-value is ≥ 0.05 (no false alarm)
- If the ML model is genuinely better, the p-value is < 0.05 (detects improvement)
- Same input + same `random_state` always gives the same p-value
- Raises a clear `ValueError` if either model doesn't support `.predict_proba()`
- Both `ml_model` and `reference_model` are user-supplied unfitted estimators

---

#### **Story 3: Understand what the p-value means**

> As a business stakeholder, I want clear documentation on what this number means and how to interpret it, so I can make yes/no decisions with confidence.

**Acceptance Criteria:**

- Documentation explains: "p < 0.05 means we have strong evidence of a real effect. A lower p-value is more convincing."
- Documentation explains: "p ≥ 0.05 means this could easily be random noise. Don't act yet."
- Documentation includes a worked example with actual numbers (pricing experiment as the motivating case)
- A data scientist can read the docstring and explain the result to a non-technical stakeholder in one sentence
- No jargon in user-facing API docs or docstring summary lines (paper terms such as "crossover TEH" and "Meinshausen" are confined to docstring `Notes` sections for researchers)

---

### Non-Goals (What We're NOT Building)

- **Subgroup discovery**: We don't recommend which model to use or auto-tune models. You bring the model spec; we validate it.
- **Effect size estimation**: We test whether an effect is real; we don't estimate its magnitude or revenue impact.
- **Causal inference**: We don't estimate "how much lift you'll gain" — just "is this real or noise?"
- **Continuous outcomes** (v1): Binary outcomes only. Continuous outcomes (revenue, time-to-event) are a v1.1 extension.
- **Non-randomised data**: Requires fully randomised two-arm assignment — `P(treatment=1 | features) = 0.5`. Bandit logs, adaptive assignment, or observational data are out of scope without IPS correction.
- **Multi-arm experiments** (v1): Exactly two arms. Three or more arms are v2.0.
- **Automatic model training**: You supply unfitted estimator specs; we fit them internally across splits.

---

## 3. Technical Specifications

### How It Works (High Level)

The module implements a peer-reviewed statistical framework (Watson & Holmes 2020) for detecting treatment effect heterogeneity with controlled false positives. Here's the one-sentence version:

> We split your data repeatedly into train and test halves, fit your model on the train half, generate predictions on the held-out half, compute a per-split test statistic, and aggregate the evidence across all splits in a way that controls false positives mathematically.

For full technical details see [Watson & Holmes (2020)](https://doi.org/10.1186/s13063-020-4076-y).

---

### Architecture Overview

```
src/samesame/
  subgroup.py             ← new module: public API + all private helpers
tests/
  test_subgroup.py        ← unit + integration tests
notebooks/
  seaquamat_replication.ipynb  ← reproducibility check (not in CI)
```

`samesame.subgroup` is **architecturally standalone** — it imports from `samesame._utils` (shared binary validation) and `samesame._types` (shared result types) but has no dependency on the distribution-shift logic (`_api.py`, `_data.py`, `_metrics.py`, `_wecdf.py`) or weighting strategies (`weights.py`).

---

### Public API

**Function 1: Test for real subgroup effects**

```python
samesame.subgroup.test_subgroup_effect(
    y,                    # binary outcome: 1 (event) or 0 (no event)
    treatment,            # binary arm assignment: 1 or 0 (must be fully randomised)
    features,             # subject features, 2D array of shape (n_subjects, n_features)
    ml_model,             # unfitted sklearn-compatible estimator with .predict_proba()
    *,                    # keyword-only after this point
    n_splits=200,         # number of balanced two-fold splits
    random_state=None,    # int | np.random.RandomState | None — for reproducibility
) -> SubgroupResult
```

**Returns** a `SubgroupResult` (extends `TestResult`):
- `.pvalue` — between 0 and 1. If < 0.05, strong evidence of real subgroup effects.
- `.statistic` — the aggregate test statistic value before p-value conversion.
- `.null_distribution` — split-wise statistics before aggregation (for researchers).

---

**Function 2: Validate ML model lift over a reference model**

```python
samesame.subgroup.test_model_lift(
    y,                    # binary outcome: 1 (event) or 0 (no event)
    treatment,            # binary arm assignment: 1 or 0
    features,             # subject features, 2D array
    ml_model,             # unfitted sklearn-compatible estimator
    reference_model,      # unfitted sklearn-compatible reference estimator
    *,
    n_splits=200,
    random_state=None,
) -> SubgroupResult
```

**Returns** a `SubgroupResult`:
- `.pvalue` — if < 0.05, the ML model captures significantly more heterogeneity than the reference.

---

### Data Requirements

| Field | Type | Constraints | Example |
|-------|------|-------------|---------|
| `y` | array-like | Binary (0 or 1) | `[0, 1, 1, 0, 1, ...]` |
| `treatment` | array-like | Binary (0 or 1); fully randomised, independent of features | `[0, 1, 1, 0, 1, ...]` |
| `features` | array-like | 2D, shape `(n_subjects, n_features)` | `[[age, income], ...]` |
| `ml_model` | object | **Unfitted** sklearn estimator with `.predict_proba()` | `RandomForestClassifier(n_estimators=100)` |
| `reference_model` | object | **Unfitted** sklearn estimator with `.predict_proba()` | `LogisticRegression()` |

**Critical constraint:** `treatment` must be **fully randomised and independent of `features`** — i.e., `P(treatment=1 | features) = 0.5` for all subjects. If arms were assigned based on subject characteristics (e.g., contextual bandits, non-random targeting), this module cannot be validly applied without IPS correction.

**Why unfitted?** Both `ml_model` and `reference_model` must be unfitted estimator specs. The function fits and refits them internally across all `n_splits` splits using `sklearn.base.clone()`. Passing a pre-fitted model would leak training data into held-out test splits and invalidate the p-value.

---

### Result Type

`SubgroupResult` extends `TestResult` (from `samesame._types`) and adds `.null_distribution`:

```python
@dataclass(frozen=True)
class SubgroupResult(TestResult):
    null_distribution: NDArray[np.float64]
```

This is consistent with `ShiftDetails` and `AdverseShiftDetails` in shape. `TestResult` provides `.statistic: float` and `.pvalue: float`.

---

### Configuration Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `n_splits` | int | 200 | Number of balanced two-fold splits. Higher → more reliable p-value, slower. |
| `random_state` | int \| RandomState \| None | None | Controls all random operations. Same value → identical p-value (bit-for-bit). |

---

### Integration Points

- **New dependencies**: None. Uses `numpy`, `scipy.stats`, `scikit-learn` — all already in samesame's dependency tree.
- **`__init__.py`**: `from . import subgroup` — exposed as `samesame.subgroup.*`. Functions are **not** hoisted to the `samesame.*` top-level namespace.
- **`pyproject.toml`**: No changes.
- **`samesame._types`**: Add `SubgroupResult` alongside existing result types.
- **Existing modules**: `test_shift`, `test_adverse_shift`, `importance_weights`, `weights` — completely untouched.
- **Docs nav**: New Tutorial page (*"Validate a pricing experiment"*) and new API reference page (`api/subgroup.md`). Update `site_description` in `mkdocs.yml`.

---

### What "Real Evidence" Means

- **p < 0.05**: Strong evidence. Less than 5% chance this happened by random noise.
- **p < 0.01**: Very strong evidence. Less than 1% chance.
- **p ≥ 0.05**: No convincing evidence. Could easily be random noise. Don't act yet.

---

### Security & Privacy

- The module accepts user-supplied model objects but does not serialise, deserialise, or transmit them.
- No network calls, no data upload, no logging of subject data or model parameters.
- You are responsible for the privacy and security of your own data and models.
- `random_state` values are not logged or transmitted.

---

### Reproducibility & Determinism

Same data + same `random_state` → always the same p-value (bit-for-bit identical). This is important for auditing and compliance.

---

### Experimental Status

`samesame.subgroup` is marked **experimental** in the module-level docstring and in the API reference page. This means the API may change before a v2.0 stability guarantee. It does not imply correctness concerns. No runtime warning is emitted at import — a docstring note is sufficient for v1.

---

## 4. Risks & Roadmap

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset too small or imbalanced → invalid test splits | Medium | High | Raise clear `ValueError` if any split has < 2 treated or < 2 control subjects. Recommend minimum n=40 with balanced arms in docs. |
| Slow performance on large datasets (n > 5,000, many features) | Medium | Low | Document recommended `n_splits` values. Parallelisation via `n_jobs` is v1.1. |
| Treatment assignment was not truly random in user's data | High | High | Front-load the randomisation requirement in docstrings and docs. No automatic detection; user responsibility. |
| Implementation diverges from paper → results don't match published benchmarks | Medium | High | Validate against SEAQUAMAT benchmark data from Watson & Holmes (2020). Provide replication notebook. |

---

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Analysts distrust p-values / need more context | High | Medium | Provide clear, non-technical interpretation. Use "strong evidence" vs. "no evidence," not "statistically significant." |
| Over-adoption without understanding the randomisation assumption | Medium | Medium | Docstrings must front-load the requirement. Raise errors on detectable violations. |

---

### Phased Rollout

#### **v1.0 (This PRD)**

**Public API (under `samesame.subgroup`):**
- `test_subgroup_effect()` — test for crossover subgroup effects
- `test_model_lift()` — test whether the ML model captures more heterogeneity than a reference

**What's included:**
- Unit tests (≥ 90% code coverage)
- Integration tests (null calibration, power checks)
- SEAQUAMAT reproducibility notebook (not in CI, not bundled)
- Clear docstrings with plain-language p-value interpretation; paper terms in `Notes` only
- Module marked **experimental** in docstring and API reference page
- Tutorial doc page: *"Validate a pricing experiment"*
- API reference doc page: `api/subgroup.md`

**What's internal only (no public API):**
- Balanced data splitting logic
- Per-split statistic computation
- Aggregate p-value construction

**Performance target:** `n_splits=200` on n=500 subjects in ≤ 60 seconds

---

#### **v1.1 (Future)**

- Parallel execution (`n_jobs` parameter, joblib backend)
- Continuous outcomes (revenue, not just binary)
- Configurable aggregation quantile (`alpha` parameter)

---

#### **v2.0 (Future)**

- Multi-arm experiments (A vs. B vs. C)
- Survival outcomes (time-to-event)
- Adaptive assignment log support (with IPS weighting)
- Public API stability guarantee + promotion from "experimental"

---

## 5. Test Automation & Validation

### Unit Tests (in `tests/test_subgroup.py`)

| Test | What It Verifies | Pass Condition |
|------|-----------------|----------------|
| `test_subgroup_effect_runs` | Executes without error on clean binary RCT data | Returns `SubgroupResult` with `.pvalue` |
| `test_subgroup_effect_reproducible` | Same `random_state` → identical p-value | Bit-for-bit match |
| `test_subgroup_effect_rejects_non_binary_y` | Non-binary outcome raises `ValueError` | Informative error message |
| `test_subgroup_effect_rejects_non_binary_treatment` | Non-binary treatment raises `ValueError` | Informative error message |
| `test_model_lift_runs` | Works with two different sklearn estimators | Returns `SubgroupResult` with `.pvalue` |
| `test_model_lift_reproducible` | Same `random_state` → identical p-value | Bit-for-bit match |
| `test_null_calibration` | 200 null datasets; p-values not concentrated near 0 | ≤ 10% of p-values < 0.05 (see AC-05 for rationale) |

### Integration Tests (Reproducibility)

| Test | Data | Target | Pass Condition |
|------|------|--------|----------------|
| **SEAQUAMAT replication** | Watson & Holmes (2020) benchmark dataset | Crossover scenario: p ≥ 0.05; Non-crossover scenario: p < 0.05 | Results match published paper |

### Regression Tests

All existing `test_api.py`, `test_iw.py`, `test_bayes.py`, `test_ood.py` pass — confirmed by running the full test suite post-implementation.

---

## 6. Acceptance Criteria

| ID | What We're Testing | Pass Condition |
|----|-------------------|----------------|
| **AC-01** | API exists and is callable | `test_subgroup_effect()` and `test_model_lift()` run without error on valid data |
| **AC-02** | p-value is reproducible | Same data + same `random_state` → identical result (bit-for-bit) |
| **AC-03** | Binary validation | Non-binary `y` or `treatment` → clear `ValueError` |
| **AC-04** | Model flexibility | Works with any sklearn-compatible unfitted estimator (LogisticRegression, RandomForest, XGBoost, etc.) |
| **AC-05** | False positive control | On 200 null datasets, ≤ 10% of p-values are < 0.05. Note: testing at 10% not 5% — a ≤ 5% bound at n=200 would fail ~50% of the time on a correctly calibrated test by pure chance. Do not tighten. |
| **AC-06** | Power (subgroup detection) | On 100 datasets with a planted crossover subgroup effect, ≥ 80% detect it (p < 0.05) |
| **AC-07** | Power (model lift) | On 100 datasets where ML model outperforms reference, ≥ 70% detect it (p < 0.05) |
| **AC-08** | No regressions | All existing samesame tests pass |
| **AC-09** | Performance | `n_splits=200`, n=500, `RandomForestClassifier(n_estimators=100)` completes in ≤ 60 seconds |
| **AC-10** | Documentation | Docstring summary lines use plain language; paper jargon ("crossover TEH", "Meinshausen") confined to `Notes` sections |

---

## Appendix: Reference & Terminology

### Key Paper

- **Watson & Holmes (2020)**: Machine learning analysis plans for randomised controlled trials. *Trials*, 21(1). [https://doi.org/10.1186/s13063-020-4076-y](https://doi.org/10.1186/s13063-020-4076-y)
  - Our implementation follows this framework for detecting treatment effect heterogeneity with type I error control. Internal terms from the paper ("crossover TEH", "non-crossover TEH", "SEAQUAMAT") are used in code comments and docstring `Notes` sections but not in public-facing documentation.

---

### Plain-Language Definitions

| Term | What It Means | Why It Matters |
|------|-------------|----------------|
| **p-value** | Probability this result happened by random chance | p < 0.05 = strong evidence (not luck). p ≥ 0.05 = could be luck. |
| **Random assignment** | Each subject has a 50/50 chance of either arm | Required for this test to be valid. If not true, results are meaningless. |
| **False positive** | Concluding a pattern is real when it was just noise | Wastes resources deploying strategies that don't work. |
| **Real subgroup** | A genuine group of subjects with different responses to the two arms | Worth building a personalised strategy for. |
| **Type I error** | Concluding something works when it doesn't | This module controls this mathematically (p < 0.05 means ≤ 5% risk). |
| **Unfitted estimator** | A model spec with hyperparameters set but not yet trained on data | Required input — the module fits it internally to prevent data leakage. |
