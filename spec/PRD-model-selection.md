---
title: "PRD: Model Lift Validation — samesame.model_selection"
version: 1.0
date_created: 2026-05-16
owner: Research Team
reference: "Watson & Holmes (2020): Machine learning analysis plans for randomised controlled trials"
tags: [prd, model-selection, rct, ab-test, feature-expansion, statistical-validation]
supersedes: spec/PRD-teh-module.md (partial)
---

# PRD: Model Lift Validation — samesame.model_selection

## 1. Executive Summary

### The Problem

**ML teams deploying personalised models from A/B experiments have no statistical proof that their model identifies treatment benefit better than a simpler baseline.**

When a data scientist builds a Random Forest to predict who benefits most from a new treatment or offer, they typically evaluate it on in-sample fit metrics. But in-sample metrics conflate general predictive quality with the ability to identify *treatment benefit specifically*. Without a formal test, there is no way to know whether the more complex estimator genuinely captures who benefits from the superior arm — or whether a logistic regression would perform just as well.

The consequences:

- Deploying complex, expensive models whose added complexity is not statistically justified
- Rejecting simpler, more interpretable models that perform equivalently (or better)
- No defensible answer to "why this model?" when stakeholders demand justification

> **Motivating example**: A pricing team builds a Random Forest to find which customers respond best to a discounted price point. The model looks good. But does it predict who benefits from the discount *better than a logistic regression*? This PRD answers exactly that question — and generalises to any two-arm experiment where you want to validate an estimator's ability to identify variation in treatment benefit.

### Proposed Solution

Implement `samesame.model_selection.test_model_lift()` — a single function that runs a peer-reviewed statistical test (Watson & Holmes 2020) to determine whether a given estimator predicts who benefits from the superior arm better than a reference estimator, with controlled false positives.

### Success Criteria

| # | Goal | How We Measure | Target |
|---|------|----------------|--------|
| SC-01 | Prevent false positives | Null calibration under controlled conditions | False positive rate ≤ 10% at n=200 null datasets |
| SC-02 | Catch real signal | Power on datasets where estimator genuinely outperforms reference | ≥ 70% power at n=400 |
| SC-03 | No regressions | Existing samesame tests still pass | 100% pass rate |
| SC-04 | Reproducible | Same data + same `random_state` = same result | Bit-for-bit identical |
| SC-05 | Fast enough | Large experiments don't time out | ≤ 60 seconds for n=500, `n_splits=200` |
| SC-06 | API simplicity | Practitioners can use it without reading the paper | Keyword-only parameters; unfitted estimators |

---

## 2. User Experience & Functionality

### User Personas

- **Data Scientist**: "I built a Random Forest to identify who benefits most from this treatment. How do I prove it beats a logistic regression — not just on fit metrics, but specifically on treatment benefit prediction?"
- **Product Manager**: "Before we invest in deploying this ML pipeline, I need statistical evidence that it outperforms the baseline. Can you give me a p-value?"

### The Core Question We Answer

**Does this estimator predict who benefits from the superior arm better than a reference estimator?**

You give the module:
- Experiment data (subject features, treatment arm, binary outcome)
- An unfitted estimator spec
- An unfitted reference estimator spec

The module returns:
- A p-value: "This difference would happen by random chance ~X% of the time"
- If p < 0.05: Your estimator predicts treatment benefit significantly better than the reference. It earns its complexity.
- If p ≥ 0.05: No convincing evidence yet. Collect more data or try a different estimator.

---

### User Stories

#### Story 1: Validate estimator lift over a reference

> As a data scientist, I want a statistical test that proves my estimator predicts who benefits from the superior arm better than a simple reference estimator — not just that it fits the data better — so I can justify deploying the more complex model to my stakeholders.

**Acceptance Criteria:**

- `samesame.model_selection.test_model_lift(y, treatment, features, estimator, reference_estimator)` runs without error
- Returns a `SubgroupResult` with `.pvalue`
- If both estimators are equivalent, the p-value is ≥ 0.05 most of the time (prevents false alarms)
- If the estimator genuinely outperforms the reference on treatment benefit prediction, the p-value is < 0.05
- Same input + same `random_state` always gives the same p-value (reproducible)
- Raises a clear `ValueError` if either estimator doesn't support `.predict_proba()`
- Raises a clear `ValueError` if `y` is not binary (0 or 1)
- Raises a clear `ValueError` if `treatment` is not binary (0 or 1)
- Both `estimator` and `reference_estimator` are user-supplied unfitted estimators
- Takes < 60 seconds for a 500-subject experiment with `n_splits=200`

#### Story 2: Communicate results to a non-technical stakeholder

> As a data scientist, I want a worked example with real numbers that I can share with a product manager, so that they can make a go/no-go deployment decision without needing to understand the statistical machinery.

**Acceptance Criteria:**

- Documentation includes a worked example (pricing experiment) showing inputs, the returned p-value, and a one-sentence conclusion a stakeholder can act on
- A data scientist can read the docstring and relay the result to a product manager in one sentence
- No jargon in user-facing API docs or docstring summary lines; paper terms ("non-crossover TEH") are confined to docstring `Notes` sections

---

### Non-Goals

- **Crossover subgroup detection**: Whether genuine crossover subgroups exist is a separate question — see `samesame.subgroup` (v1.1).
- **Effect size estimation**: We test whether the lift is real; we do not estimate its magnitude or revenue impact.
- **General model quality**: This test is specifically about treatment-benefit prediction, not predictive quality in general. Passing this test does not mean the estimator is well-calibrated or useful for non-RCT tasks.
- **Continuous outcomes** (v1): Binary outcomes only. Revenue, time-to-event outcomes are v1.1.
- **Non-randomised data**: Requires fully randomised two-arm assignment — `P(treatment=1 | features) = 0.5`. Bandit logs and adaptive assignment are out of scope without IPS correction.
- **Multi-arm experiments** (v1): Exactly two arms.
- **Automatic model selection**: You supply unfitted estimator specs; we validate them. We do not recommend which model to use.

---

## 3. Technical Specifications

### How It Works

We split your data repeatedly into train and test halves. On each split, we fit both your estimator and the reference estimator on the train half, generate predictions on the held-out half, compute a per-split test statistic comparing their ability to capture variation in treatment benefit, and aggregate the evidence across all splits in a way that controls false positives mathematically.

Concretely: for each split, the per-split statistic measures whether the estimator's predicted scores separate the treatment benefit signal more than the reference estimator's predicted scores. The aggregate p-value is `min(1, Q_alpha({2*p_i}))` where `Q_alpha` is the alpha-quantile (default alpha=0.5, the median) over split-wise p-values.

For full technical details see [Watson & Holmes (2020)](https://doi.org/10.1186/s13063-020-4076-y). The method corresponds to the non-crossover TEH test in that paper.

---

### Architecture Overview

```
src/samesame/
  model_selection.py    ← test_model_lift: public API + private helpers
tests/
  test_model_selection.py
```

`samesame.model_selection` imports from `samesame._utils` (binary validation) and may reuse `TestResult` from an owning public seam such as `samesame.shift`. If a shared `SubgroupResult` becomes necessary once both RCT validation modules exist, extract it then rather than depending on a pre-emptive `_types` module. No dependency on distribution-shift logic in `shift.py`, weighting strategies (`weights.py`), or `samesame.subgroup`.

---

### Public API

```python
samesame.model_selection.test_model_lift(
    y,                    # binary outcome: 1 (event) or 0 (no event)
    treatment,            # binary arm assignment: 1 or 0 (must be fully randomised)
    features,             # subject features, 2D array of shape (n_subjects, n_features)
    estimator,            # unfitted sklearn-compatible estimator with .predict_proba()
    reference_estimator,  # unfitted sklearn-compatible reference estimator with .predict_proba()
    *,                    # keyword-only after this point
    n_splits=200,         # number of balanced two-fold splits
    random_state=None,    # int | np.random.RandomState | None — for reproducibility
) -> SubgroupResult
```

**Returns** a `SubgroupResult`:
- `.pvalue` — between 0 and 1. If < 0.05, strong evidence the estimator predicts treatment benefit better than the reference.
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
| `reference_estimator` | object | **Unfitted** sklearn-compatible estimator with `.predict_proba()` | `LogisticRegression()` |

**Critical constraint:** `treatment` must be **fully randomised and independent of `features`** — i.e., `P(treatment=1 | features) = 0.5` for all subjects.

**Why unfitted?** Both estimators must be unfitted specs. The function fits and refits them internally across all `n_splits` splits using `sklearn.base.clone()`. Passing a pre-fitted model would leak training data into held-out test splits and invalidate the p-value.

---

### Result Type

`SubgroupResult` (extends `TestResult`):

```python
@dataclass(frozen=True)
class SubgroupResult(TestResult):
    null_distribution: NDArray[np.float64]
```

`TestResult` provides `.statistic: float` and `.pvalue: float`. `SubgroupResult` is shared with `samesame.subgroup`, but its final home should be chosen only once both modules exist as real callers.

---

### Configuration Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `n_splits` | int | 200 | Number of balanced two-fold splits. Higher → more reliable p-value, slower. |
| `random_state` | int \| RandomState \| None | None | Controls all random operations. Same value → identical p-value. |

---

### Integration Points

- **`__init__.py`**: `from . import model_selection` — exposed as `samesame.model_selection.*`. Functions are **not** hoisted to the `samesame.*` top-level namespace.
- **Shared result type**: Reuse `TestResult` from an owning public seam such as `samesame.shift`. Introduce a shared `SubgroupResult` only when both RCT validation modules exist and genuinely need the same public type.
- **`pyproject.toml`**: No changes. Uses `numpy`, `scipy.stats`, `scikit-learn` — already in the dependency tree.
- **Existing modules**: `test_shift`, `test_adverse_shift`, `importance_weights`, `weights` — completely untouched.
- **Docs nav**: New Tutorial page (*"Validate a pricing experiment"*) and API reference page (`api/model_selection.md`). Update `site_description` in `mkdocs.yml`.

---

### Security & Privacy

- The module accepts user-supplied model objects but does not serialise, deserialise, or transmit them.
- No network calls, no data upload, no logging of subject data or model parameters.
- `random_state` values are not logged or transmitted.

---

### Reproducibility & Determinism

Same data + same `random_state` → always the same p-value (bit-for-bit identical). Required for auditing and compliance.

---

### Experimental Status

`samesame.model_selection` is marked **experimental** in the module-level docstring and in `api/model_selection.md`. No runtime warning is emitted at import — a docstring note is sufficient. "Experimental" means the API may change before a v2.0 stability guarantee; it does not imply correctness concerns.

---

## 4. Risks & Roadmap

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset too small or imbalanced → invalid test splits | Medium | High | Raise `ValueError` if any split has < 2 treated or < 2 control. Document minimum n=40 with balanced arms. |
| Implementation diverges from paper → results don't match published benchmarks | Medium | High | Validate against SEAQUAMAT benchmark data from Watson & Holmes (2020). Provide replication notebook. |
| Treatment assignment was not truly random | High | High | Front-load the randomisation requirement in docstrings. No automatic detection; user responsibility. |
| Slow performance on large datasets | Medium | Low | Document recommended `n_splits` values. Parallelisation via `n_jobs` is v1.1. |

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Practitioners interpret p ≥ 0.05 as "the reference is better" | Medium | Medium | Docstrings must clarify: p ≥ 0.05 means no convincing evidence either way — not that the reference wins. |
| Over-adoption without understanding the randomisation assumption | Medium | Medium | Docstrings must front-load the requirement. |

---

### Phased Rollout

#### v1.0 (This PRD)

**Public API:**
- `samesame.model_selection.test_model_lift()` — test whether estimator predicts treatment benefit better than a reference

**What's included:**
- Unit tests (≥ 90% code coverage)
- Integration tests (null calibration, power checks)
- SEAQUAMAT reproducibility notebook (not in CI)
- Shared `SubgroupResult` only if both RCT validation modules exist and genuinely need the same public type
- Clear docstrings with plain-language p-value interpretation; paper terms in `Notes` only
- Module marked **experimental**
- Tutorial: *"Validate a pricing experiment"*
- API reference: `api/model_selection.md`

**What's internal only:**
- Balanced data splitting logic
- Per-split statistic computation
- Aggregate p-value construction

**Performance target:** `n_splits=200` on n=500 subjects in ≤ 60 seconds

---

#### v1.1 (Future)

- `samesame.subgroup.test_subgroup_effect()` — separate PRD
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

### Unit Tests (`tests/test_model_selection.py`)

| Test | What It Verifies | Pass Condition |
|------|-----------------|----------------|
| `test_model_lift_runs` | Executes without error on clean binary RCT data with two estimators | Returns `SubgroupResult` with `.pvalue` |
| `test_model_lift_reproducible` | Same `random_state` → identical p-value | Bit-for-bit match |
| `test_model_lift_rejects_non_binary_y` | Non-binary outcome raises `ValueError` | Informative error message |
| `test_model_lift_rejects_non_binary_treatment` | Non-binary treatment raises `ValueError` | Informative error message |
| `test_model_lift_rejects_no_predict_proba` | Estimator without `.predict_proba()` raises `ValueError` | Informative error message |
| `test_null_calibration` | 200 null datasets; p-values not concentrated near 0 | ≤ 10% of p-values < 0.05 |
| `test_power` | 100 datasets where estimator outperforms reference | ≥ 70% detect it (p < 0.05) |

### Regression Tests

All existing `test_api.py`, `test_iw.py`, `test_bayes.py`, `test_ood.py` pass — confirmed by running the full test suite post-implementation.

---

## 6. Acceptance Criteria

| ID | What We're Testing | Pass Condition |
|----|-------------------|----------------|
| **AC-01** | API exists and is callable | `test_model_lift()` runs without error on valid data |
| **AC-02** | Reproducible | Same data + same `random_state` → identical result (bit-for-bit) |
| **AC-03** | Binary validation | Non-binary `y` or `treatment` → clear `ValueError` |
| **AC-04** | Estimator validation | Estimator without `.predict_proba()` → clear `ValueError` |
| **AC-05** | Model flexibility | Works with any sklearn-compatible unfitted estimator pair |
| **AC-06** | False positive control | On 200 null datasets, ≤ 10% of p-values are < 0.05. Note: testing at 10% not 5% — a ≤ 5% bound at n=200 would fail ~50% of the time on a correctly calibrated test. Do not tighten. |
| **AC-07** | Power | On 100 datasets where the estimator outperforms the reference, ≥ 70% detect it (p < 0.05) |
| **AC-08** | No regressions | All existing samesame tests pass |
| **AC-09** | Performance | `n_splits=200`, n=500, `RandomForestClassifier(n_estimators=100)` + `LogisticRegression()` completes in ≤ 60 seconds |
| **AC-10** | Documentation | Docstring summary lines use plain language; paper jargon ("non-crossover TEH") confined to `Notes` sections |

---

## Appendix: Reference & Terminology

### Key Paper

- **Watson & Holmes (2020)**: Machine learning analysis plans for randomised controlled trials. *Trials*, 21(1). [https://doi.org/10.1186/s13063-020-4076-y](https://doi.org/10.1186/s13063-020-4076-y)
  - Our implementation follows the non-crossover TEH test in this framework. Paper terms ("non-crossover TEH", "SEAQUAMAT") are used in code comments and docstring `Notes` sections but not in public-facing documentation.

### Plain-Language Definitions

| Term | What It Means |
|------|-------------|
| **p-value** | Probability this result happened by random chance. p < 0.05 = strong evidence (not luck). p ≥ 0.05 = could be luck, or not enough data. |
| **Random assignment** | Each subject has a 50/50 chance of either arm. Required for this test to be valid. |
| **Unfitted estimator** | A model spec with hyperparameters set but not yet trained on data. The module fits it internally. |
| **Reference estimator** | A simpler baseline model (e.g. logistic regression). The test asks whether `estimator` beats `reference_estimator` at predicting treatment benefit. |
| **Treatment benefit** | How much better a subject does on the superior arm compared to the other arm. The test measures whether the estimator captures variation in this quantity. |
