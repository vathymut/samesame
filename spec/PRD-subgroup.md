---
title: "PRD: Subgroup Effect Validation — samesame.subgroup"
version: 1.0
date_created: 2026-05-16
owner: Research Team
reference: "Watson & Holmes (2020): Machine learning analysis plans for randomised controlled trials"
tags: [prd, subgroup, rct, ab-test, feature-expansion, statistical-validation]
supersedes: spec/PRD-teh-module.md (partial)
---

# PRD: Subgroup Effect Validation — samesame.subgroup

## 1. Executive Summary

### The Problem

**ML teams running A/B experiments have no way to validate whether their model found real crossover subgroups or statistical noise.**

When teams run two-arm experiments, data scientists often build ML models to find subgroups that respond *differently in direction* to the two conditions — where some subjects do better on arm A and others do better on arm B. Without formal statistical validation, there is no way to know if those patterns are real or just noise. The consequences:

- Deploying differentiated strategies based on spurious patterns (false positives)
- Missing real, profitable heterogeneity because we conflate luck with signal
- Undermining trust in the experimentation process across the organisation

> **Motivating example**: A product team builds a classifier to find customers who prefer product variant A over variant B — not just that one variant is universally better, but that genuinely different segments respond better to different variants. The model looks compelling. But do those crossover preferences actually exist, or did the model find a spurious pattern? This PRD answers exactly that question.

### Proposed Solution

Implement `samesame.subgroup.test_subgroup_effect()` — a single function that tests whether real crossover subgroups exist in a two-arm experiment, with controlled false positives.

This is the complementary tool to `samesame.model_selection.test_model_lift()` (v1.0), which tests non-crossover heterogeneity (degree of benefit variation). These are independent tools — use `test_subgroup_effect` when the question is whether the *direction* of the treatment effect genuinely differs across subjects.

### Success Criteria

| # | Goal | How We Measure | Target |
|---|------|----------------|--------|
| SC-01 | Prevent false positives | Null calibration under controlled conditions | False positive rate ≤ 10% at n=200 null datasets |
| SC-02 | Catch real signals | Power on datasets with planted crossover subgroup | ≥ 80% power at n=400 |
| SC-03 | No regressions | Existing samesame tests still pass | 100% pass rate |
| SC-04 | Reproducible | Same data + same `random_state` = same result | Bit-for-bit identical |
| SC-05 | Fast enough | Large experiments don't time out | ≤ 60 seconds for n=500, `n_splits=200` |
| SC-06 | API simplicity | Easy for practitioners to use | Keyword-only parameters; unfitted estimator |

---

## 2. User Experience & Functionality

### User Personas

- **Data Scientist / Analyst**: "I built a model that finds subgroups where different arms win for different groups. How do I prove those subgroups are real — not just overfitting?"
- **Business Stakeholder**: "Should we run a differentiated strategy for different segments? Is this experiment result safe to act on, or are we chasing noise?"

### The Core Question We Answer

**Do real crossover subgroups exist — or is this pattern just noise?**

You give the module:
- Experiment data (subject features, treatment arm, binary outcome)
- An unfitted estimator spec

The module returns:
- A p-value: "This result would happen by random chance ~X% of the time"
- If p < 0.05: Strong evidence of real crossover subgroups. A differentiated strategy is statistically justified.
- If p ≥ 0.05: No convincing evidence. Do not build a differentiated strategy yet.

---

### User Stories

#### Story 1: Test for real crossover subgroup effects

> As a data scientist, I want to validate that my estimator found actual subgroups with different treatment responses — not just patterns from random noise — so that I can confidently recommend a differentiated strategy to leadership.

**Acceptance Criteria:**

- `samesame.subgroup.test_subgroup_effect(y, treatment, features, estimator)` runs without error
- Returns a `SubgroupResult` with `.pvalue`
- If I pass null (random) data, the p-value is ≥ 0.05 most of the time (prevents false alarms)
- If I pass data with a real crossover subgroup, the p-value is < 0.05 (catches real signals)
- Same input + same `random_state` always gives the same p-value (reproducible)
- Raises a clear `ValueError` if `treatment` is not binary (0 or 1)
- Raises a clear `ValueError` if `y` is not binary (0 or 1)
- Takes < 60 seconds for a 500-subject experiment with `n_splits=200`
- Works with any sklearn-compatible unfitted estimator

#### Story 2: Communicate results to a non-technical stakeholder

> As a data scientist, I want a worked example with real numbers that I can share with a business stakeholder, so that they can make a yes/no decision on a differentiated strategy without needing to understand the statistical machinery.

**Acceptance Criteria:**

- Documentation includes a worked example with actual numbers showing inputs, the returned p-value, and a one-sentence conclusion a stakeholder can act on
- A data scientist can read the docstring and relay the result to a non-technical stakeholder in one sentence
- No jargon in user-facing API docs or docstring summary lines; paper terms ("crossover TEH") are confined to docstring `Notes` sections

---

### Non-Goals

- **Model comparison**: Whether one estimator beats another for treatment-benefit prediction is a separate question — see `samesame.model_selection.test_model_lift()` (v1.0).
- **Effect size estimation**: We test whether crossover subgroups are real; we do not estimate their size or revenue impact.
- **Non-crossover heterogeneity**: When one arm is universally superior but the *degree* of benefit varies by subject, use `samesame.model_selection.test_model_lift()` instead. This function tests crossover effects only.
- **Continuous outcomes** (v1): Binary outcomes only.
- **Non-randomised data**: Requires fully randomised two-arm assignment — `P(treatment=1 | features) = 0.5`. Bandit logs and adaptive assignment are out of scope without IPS correction.
- **Multi-arm experiments** (v1): Exactly two arms.
- **Subgroup discovery**: We validate whether subgroups are real; we do not recommend which model to use or auto-tune models.

---

## 3. Technical Specifications

### How It Works

We split your data repeatedly into train and test halves. On each split, we fit your estimator on the train half, generate predictions on the held-out half, compute a per-split test statistic quantifying evidence for crossover treatment effect heterogeneity, and aggregate the evidence across all splits in a way that controls false positives mathematically.

Concretely: for each split, the per-split statistic measures whether the estimator's predicted scores identify subjects for whom the arm ordering reverses. The aggregate p-value is `min(1, Q_alpha({2*p_i}))` where `Q_alpha` is the alpha-quantile (default alpha=0.5, the median) over split-wise p-values.

For full technical details see [Watson & Holmes (2020)](https://doi.org/10.1186/s13063-020-4076-y). The method corresponds to the crossover TEH test in that paper.

---

### Architecture Overview

```
src/samesame/
  subgroup.py           ← test_subgroup_effect: public API + private helpers
tests/
  test_subgroup.py
```

`samesame.subgroup` imports from `samesame._utils` (binary validation) and `samesame._types` (`SubgroupResult`). No dependency on distribution-shift logic (`_api.py`, `_data.py`, `_metrics.py`, `_wecdf.py`), weighting strategies (`weights.py`), or `samesame.model_selection`.

---

### Public API

```python
samesame.subgroup.test_subgroup_effect(
    y,                    # binary outcome: 1 (event) or 0 (no event)
    treatment,            # binary arm assignment: 1 or 0 (must be fully randomised)
    features,             # subject features, 2D array of shape (n_subjects, n_features)
    estimator,            # unfitted sklearn-compatible estimator with .predict_proba()
    *,                    # keyword-only after this point
    n_splits=200,         # number of balanced two-fold splits
    random_state=None,    # int | np.random.RandomState | None — for reproducibility
) -> SubgroupResult
```

**Returns** a `SubgroupResult`:
- `.pvalue` — between 0 and 1. If < 0.05, strong evidence of real crossover subgroups.
- `.statistic` — the aggregate test statistic value before p-value conversion.
- `.null_distribution` — split-wise statistics before aggregation (for researchers).

---

### Data Requirements

| Field | Type | Constraints | Example |
|-------|------|-------------|---------|
| `y` | array-like | Binary (0 or 1) | `[0, 1, 1, 0, 1, ...]` |
| `treatment` | array-like | Binary (0 or 1); fully randomised, independent of features | `[0, 1, 1, 0, 1, ...]` |
| `features` | array-like | 2D, shape `(n_subjects, n_features)` | `[[age, income], ...]` |
| `estimator` | object | **Unfitted** sklearn-compatible estimator with `.predict_proba()` | `RandomForestClassifier(n_estimators=100)` |

**Critical constraint:** `treatment` must be **fully randomised and independent of `features`** — i.e., `P(treatment=1 | features) = 0.5` for all subjects.

**Why unfitted?** The estimator must be an unfitted spec. The function fits and refits it internally across all `n_splits` splits using `sklearn.base.clone()`. Passing a pre-fitted model would leak training data into held-out test splits and invalidate the p-value.

---

### Result Type

`SubgroupResult` (defined in `samesame._types`, extends `TestResult`):

```python
@dataclass(frozen=True)
class SubgroupResult(TestResult):
    null_distribution: NDArray[np.float64]
```

`TestResult` provides `.statistic: float` and `.pvalue: float`. `SubgroupResult` is shared with `samesame.model_selection` and defined once in `_types.py`.

---

### Configuration Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `n_splits` | int | 200 | Number of balanced two-fold splits. Higher → more reliable p-value, slower. |
| `random_state` | int \| RandomState \| None | None | Controls all random operations. Same value → identical p-value. |

---

### Integration Points

- **`__init__.py`**: `from . import subgroup` — exposed as `samesame.subgroup.*`. Functions are **not** hoisted to the `samesame.*` top-level namespace.
- **`samesame._types`**: `SubgroupResult` is already defined in `_types.py` from v1.0. No change needed.
- **`pyproject.toml`**: No changes. Uses `numpy`, `scipy.stats`, `scikit-learn` — already in the dependency tree.
- **Existing modules**: `test_shift`, `test_adverse_shift`, `importance_weights`, `weights`, `test_model_lift` — completely untouched.
- **Docs nav**: New Tutorial page (*"Detect subgroup treatment effects"*) and API reference page (`api/subgroup.md`). Update `site_description` in `mkdocs.yml` if not already updated in v1.0.

---

### Security & Privacy

- The module accepts user-supplied model objects but does not serialise, deserialise, or transmit them.
- No network calls, no data upload, no logging of subject data or model parameters.

---

### Reproducibility & Determinism

Same data + same `random_state` → always the same p-value (bit-for-bit identical). Required for auditing and compliance.

---

### Experimental Status

`samesame.subgroup` is marked **experimental** in the module-level docstring and in `api/subgroup.md`. No runtime warning is emitted at import. "Experimental" means the API may change before a v2.0 stability guarantee; it does not imply correctness concerns.

---

## 4. Risks & Roadmap

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset too small or imbalanced → invalid test splits | Medium | High | Raise `ValueError` if any split has < 2 treated or < 2 control. Document minimum n=40. |
| Implementation diverges from paper → results don't match published benchmarks | Medium | High | Validate against SEAQUAMAT benchmark data from Watson & Holmes (2020). Provide replication notebook. |
| Treatment assignment was not truly random | High | High | Front-load the randomisation requirement in docstrings. No automatic detection; user responsibility. |
| Slow performance on large datasets | Medium | Low | Document recommended `n_splits` values. Parallelisation via `n_jobs` is v1.2. |

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Practitioners conflate crossover and non-crossover heterogeneity | Medium | Medium | Docstrings and docs must clearly distinguish: `test_subgroup_effect` tests for *direction* reversal; `test_model_lift` tests for *degree* variation. |
| Over-adoption without understanding the randomisation assumption | Medium | Medium | Docstrings must front-load the requirement. |

---

### Phased Rollout

#### v1.1 (This PRD)

**Public API:**
- `samesame.subgroup.test_subgroup_effect()` — test for crossover subgroup effects

**What's included:**
- Unit tests (≥ 90% code coverage)
- Integration tests (null calibration, power checks)
- SEAQUAMAT reproducibility notebook (not in CI)
- Clear docstrings with plain-language p-value interpretation; paper terms in `Notes` only
- Module marked **experimental**
- Tutorial: *"Detect subgroup treatment effects"*
- API reference: `api/subgroup.md`

**What's internal only:**
- Balanced data splitting logic
- Per-split statistic computation
- Aggregate p-value construction

**Dependency:** `SubgroupResult` is already in `samesame._types` from v1.0. No new shared types needed.

**Performance target:** `n_splits=200` on n=500 subjects in ≤ 60 seconds

---

#### v1.2 (Future)

- Parallel execution (`n_jobs` parameter, joblib backend)
- Continuous outcomes (revenue, not just binary)

---

#### v2.0 (Future)

- Multi-arm experiments (A vs. B vs. C)
- Survival outcomes (time-to-event)
- Adaptive assignment log support (with IPS weighting)
- Public API stability guarantee + promotion from "experimental"

---

## 5. Test Automation & Validation

### Unit Tests (`tests/test_subgroup.py`)

| Test | What It Verifies | Pass Condition |
|------|-----------------|----------------|
| `test_subgroup_effect_runs` | Executes without error on clean binary RCT data | Returns `SubgroupResult` with `.pvalue` |
| `test_subgroup_effect_reproducible` | Same `random_state` → identical p-value | Bit-for-bit match |
| `test_subgroup_effect_rejects_non_binary_y` | Non-binary outcome raises `ValueError` | Informative error message |
| `test_subgroup_effect_rejects_non_binary_treatment` | Non-binary treatment raises `ValueError` | Informative error message |
| `test_null_calibration` | 200 null datasets; p-values not concentrated near 0 | ≤ 10% of p-values < 0.05 |
| `test_power` | 100 datasets with planted crossover subgroup effect | ≥ 80% detect it (p < 0.05) |

### Regression Tests

All existing `test_api.py`, `test_iw.py`, `test_bayes.py`, `test_ood.py`, and `test_model_selection.py` pass — confirmed by running the full test suite post-implementation.

---

## 6. Acceptance Criteria

| ID | What We're Testing | Pass Condition |
|----|-------------------|----------------|
| **AC-01** | API exists and is callable | `test_subgroup_effect()` runs without error on valid data |
| **AC-02** | Reproducible | Same data + same `random_state` → identical result (bit-for-bit) |
| **AC-03** | Binary validation | Non-binary `y` or `treatment` → clear `ValueError` |
| **AC-04** | Model flexibility | Works with any sklearn-compatible unfitted estimator |
| **AC-05** | False positive control | On 200 null datasets, ≤ 10% of p-values are < 0.05. Note: testing at 10% not 5% — a ≤ 5% bound at n=200 would fail ~50% of the time on a correctly calibrated test. Do not tighten. |
| **AC-06** | Power (subgroup detection) | On 100 datasets with a planted crossover subgroup effect, ≥ 80% detect it (p < 0.05) |
| **AC-07** | No regressions | All existing samesame tests pass, including `test_model_selection.py` |
| **AC-08** | Performance | `n_splits=200`, n=500, `RandomForestClassifier(n_estimators=100)` completes in ≤ 60 seconds |
| **AC-09** | Documentation | Docstring summary lines use plain language; paper jargon ("crossover TEH") confined to `Notes` sections |

---

## Appendix: Reference & Terminology

### Key Paper

- **Watson & Holmes (2020)**: Machine learning analysis plans for randomised controlled trials. *Trials*, 21(1). [https://doi.org/10.1186/s13063-020-4076-y](https://doi.org/10.1186/s13063-020-4076-y)
  - Our implementation follows the crossover TEH test in this framework. Paper terms ("crossover TEH", "SEAQUAMAT") are used in code comments and docstring `Notes` sections but not in public-facing documentation.

### Plain-Language Definitions

| Term | What It Means |
|------|-------------|
| **p-value** | Probability this result happened by random chance. p < 0.05 = strong evidence. p ≥ 0.05 = could be luck, or not enough data. |
| **Crossover subgroup** | A group of subjects for whom arm B is better, while arm A is better for the rest. This is what this test detects. |
| **Random assignment** | Each subject has a 50/50 chance of either arm. Required for this test to be valid. |
| **Unfitted estimator** | A model spec with hyperparameters set but not yet trained on data. The module fits it internally to prevent data leakage. |
| **False positive** | Concluding a crossover pattern is real when it was just noise. This module controls this mathematically. |
