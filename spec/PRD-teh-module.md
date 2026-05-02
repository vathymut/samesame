---
title: "PRD: Pricing Experiment Testing Module — samesame.subgroup"
version: 1.0
date_created: 2026-05-01
owner: Research Team
reference: "Watson & Holmes (2020): Machine learning analysis plans for randomised controlled trials"
tags: [prd, subgroup, pricing, ab-test, feature-expansion, statistical-validation]
---

# PRD: Pricing Experiment Testing Module

## 1. Executive Summary

### The Problem (Business View)

**Our pricing team can't trust their experiments.**

When we run A/B tests with different prices, our data scientists often build machine learning models to find customer segments that respond differently to pricing. But we have no way to know if the patterns they find are *real* or just **statistical noise**. Without statistical validation, we risk:

- Deploying pricing strategies that only *appeared* to work in the test data (false positives)
- Missing real, profitable pricing opportunities because we conflate luck with signal
- Undermining trust in the experimentation process across the organization

**We're deploying personalized pricing strategies with no statistical validation.**

A personalized pricing model might look good on historical data — but we don't know if it actually would work better than a simple flat-price policy. This is a $M-level risk.

### Proposed Solution

Implement a `samesame.subgroup` module that answers two concrete questions about pricing experiment data:

1. **Do we have real price-sensitive customer segments?** (Not just random noise)
2. **Does a personalized pricing model actually beat our baseline flat pricing?** (With statistical proof)

The module uses a rigorous, peer-reviewed statistical framework (Watson & Holmes 2020) that prevents false positives by design. **You don't need to understand the statistics** — the API is simple. You provide experiment data + your model, and get a p-value that tells you: *"This result would happen by chance ~X% of the time."*

### Success Criteria

| # | Business Goal | How We Measure | Target |
|---|---------------|----------------|--------|
| SC-01 | Prevent false positives | Statistical validation under controlled conditions | False positive rate ≤ 5% |
| SC-02 | Catch real signals | Ability to detect real customer segments | Detect 80%+ of true segments at n=400 |
| SC-03 | No regressions | Existing samesame tests still pass | 100% pass rate |
| SC-04 | Reproducible results | Same data + same settings = same result | Always bit-for-bit identical |
| SC-05 | Fast enough | Large experiments don't time out | ≤ 60 seconds for n=500 |
| SC-06 | API simplicity | Easy for practitioners to use | Keyword-only, unfitted estimator |

---

## 2. User Experience & Functionality

### User Personas

- **Business Stakeholder / Pricing Manager**: "I need to know if this experiment is safe to act on. Can we trust the conclusions?"
- **Product Manager**: "I need to know if the data science team's claim is valid before we allocate engineering resources to deploy a personalized pricing strategy."
- **Data Scientist / Pricing Analyst**: "I built a model that finds price-sensitive segments. How do I prove it's not just overfitting?"

---

### The Core Questions We Answer

This module answers exactly **two** questions:

#### **Question 1: Are there real price-sensitive customer segments in our experiment?**
*Or did we just get lucky with noise in the data?*

You give the module:
- Experiment data (customer features, which price arm they saw, whether they bought)
- Your trained ML model (Random Forest, XGBoost, whatever)

The module returns:
- A p-value: "This result would happen by random chance ~X% of the time"
- If p < 0.05: You have evidence of real segments. Deploy with confidence.
- If p ≥ 0.05: This looks like noise. Don't deploy.

---

#### **Question 2: Does a personalized pricing model actually beat flat pricing?**
*Not just "this model does well on the training data," but "this model would win in a fresh experiment"?*

You give the module:
- Experiment data
- Your personalized pricing model
- A baseline (flat-price) pricing model

The module returns:
- A p-value: "This difference would happen by random chance ~X% of the time"
- If p < 0.05: Your personalized model is genuinely better. Deploy it.
- If p ≥ 0.05: You don't have evidence yet. Collect more data or try a different model.

---

### User Stories

#### **Story 1: Test for real price-sensitive segments**

> As a pricing analyst, I want to validate that my ML model found actual price-sensitive customer segments — not just patterns from random noise — so that I can confidently recommend a segment-based pricing strategy to leadership.

**Acceptance Criteria:**

- `test_segments(y, pricing_arm, customer_features, my_model)` runs without error
- Returns a result object with `.pvalue` 
- If I pass random/noise data, the p-value is ≥ 0.05 most of the time (prevents false alarms)
- If I pass data with a **real** segment (e.g., one group buys more at high prices, another at low prices), the p-value is < 0.05 (catches real signals)
- Same input + same random seed always gives the same p-value (reproducible)
- Raises a clear error if the pricing arm is not binary (0 or 1)
- Raises a clear error if the outcome is not binary (0 or 1)
- Takes < 60 seconds for a 500-customer experiment
- Works with any sklearn-compatible model (Random Forest, XGBoost, etc.)

---

#### **Story 2: Validate a personalized pricing model**

> As a product manager, I want a statistical test that proves our personalized pricing model would actually beat our flat-price baseline in a new experiment — not just that it looks good on the data we trained it on — so I can confidently approve the engineering effort to deploy it.

**Acceptance Criteria:**

- `test_model_improvement(y, pricing_arm, customer_features, personalized_model, baseline_model)` runs without error
- Returns a result object with `.pvalue`
- If both models are equally good, the p-value is ≥ 0.05 (no false alarm)
- If the personalized model is genuinely better, the p-value is < 0.05 (detects improvement)
- Same input + same random seed always gives the same p-value
- Raises a clear error if either model doesn't support probability predictions
- Works with any sklearn-compatible models
- Baseline and personalized models are both user-supplied (not auto-trained)

---

#### **Story 3: Understand what the p-value means**

> As a business stakeholder, I want clear documentation on what this number means and how to interpret it, so I can make yes/no decisions with confidence.

**Acceptance Criteria:**

- Documentation explains: "p < 0.05 means we have strong evidence of a real effect. A lower p-value is more convincing."
- Documentation explains: "p ≥ 0.05 means this could easily be random noise. Don't deploy yet."
- Documentation includes a real example with actual numbers
- A data scientist can read the docstring and explain the result to a non-technical stakeholder in 1 sentence
- No mention of "Meinshausen," "crossover TEH," "ANOVA," or other jargon in the public API docs

---

### Non-Goals (What We're NOT Building)

- **Segment discovery**: We don't tell you which model to use or auto-tune models. You bring the model; we validate it.
- **Revenue optimization**: We test if a model works; we don't optimize pricing levels or experiment design.
- **Causal inference**: We don't estimate "how much revenue you'll gain" — just "is this real or noise?"
- **Continuous outcomes** (v1): We only test binary purchases (bought or not). Revenue/margin is a v1.1 extension.
- **Non-randomized data**: Our method requires a fair coin flip (50/50 random assignment to pricing arms). Bandit logs or adaptive pricing data are out of scope.
- **Multi-arm experiments** (v1): Only two pricing arms (Arm A vs. Arm B). 3+ arms is v2.0.
- **Automatic model training**: You bring trained models; we don't train them for you.

---

## 3. Technical Specifications

### How It Works (High Level)

The module uses a peer-reviewed statistical method (Watson & Holmes 2020) to prevent false positives. **You don't need to understand the details.** Here's the one-sentence version:

> We split your data repeatedly into training and test sets, make predictions on the held-out test data, and combine the evidence in a way that controls false positives mathematically.

If you want the full details: see the [Watson & Holmes (2020)](https://doi.org/10.1186/s13063-020-4076-y) paper.

---

### Architecture Overview

```
samesame/
  subgroup.py             ← new module (public API)
  _subgroup_internals.py  ← private utilities
tests/
  test_subgroup.py        ← unit + integration tests
notebooks/
  seaquamat_replication.ipynb  ← reproducibility check
```

The `samesame.subgroup` module is **standalone** — it doesn't depend on the distribution-shift or weighting code. It only uses `numpy`, `scipy`, and `scikit-learn` (all already in samesame's dependency tree).

---

### Public API

**Function 1: Test for real price-sensitive segments**

```python
samesame.subgroup.test_segments(
    y,                        # purchase (1) or not (0)
    pricing_arm,              # arm 0 or arm 1 (must be random 50/50)
    customer_features,        # customer data (2D array)
    your_model,               # fitted ML model (sklearn-compatible)
    num_splits=200,           # internal: how many train/test splits
    random_seed=None,         # for reproducibility
) -> Result
```

**Returns:**
- `.pvalue` — a number between 0 and 1. If < 0.05, you have strong evidence of real segments.
- `.null_distribution` — internal: the raw test results before aggregation (for researchers)

---

**Function 2: Validate a personalized pricing model**

```python
samesame.subgroup.test_model_improvement(
    y,                        # purchase (1) or not (0)
    pricing_arm,              # arm 0 or arm 1
    customer_features,        # customer data
    personalized_model,       # your ML model (sklearn-compatible)
    baseline_model,           # flat-price or simple baseline (sklearn-compatible)
    num_splits=200,           # internal: how many train/test splits
    random_seed=None,
) -> Result
```

**Returns:**
- `.pvalue` — if < 0.05, your personalized model is genuinely better than the baseline.

---

### Data Requirements

| Field | Type | Constraints | Example |
|-------|------|-------------|---------|
| `y` | array | Binary (0 or 1) | [0, 1, 1, 0, 1, ...] |
| `pricing_arm` | array | Binary (0 or 1); must be random 50/50 | [0, 1, 1, 0, 1, ...] |
| `customer_features` | array | 2D shape (n_customers, n_features) | [[age, income], [35, 50000], ...] |
| `your_model` | object | Fitted sklearn model with `.predict_proba()` | `RandomForestClassifier()` |
| `baseline_model` | object | Fitted sklearn model with `.predict_proba()` | `LogisticRegression()` |

**Key constraint:** Pricing arms must be **fully randomized and independent of customer features**. If you assigned higher prices to wealthy customers (non-random), this module cannot be used without special correction. Tell us if that's your situation.

---

### Configuration Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|------------|
| `num_splits` | int | 200 | How many internal train/test splits to use. Higher = more reliable p-value but slower. |
| `random_seed` | int | None | For reproducibility. If you pass the same seed, you get the same p-value. |

---

### Integration Points

- **Dependencies**: `numpy`, `scipy.stats`, `scikit-learn` — all already in samesame
- **New dependencies**: None
- **`__init__.py`**: `samesame.subgroup` is added as a public module re-export
- **`pyproject.toml`**: No changes
- **Existing modules**: `test_shift`, `importance_weights`, `advanced` — completely untouched

---

### What "Real Evidence" Means

- **p < 0.05**: Strong evidence. Less than 5% chance this happened by random noise.
- **p < 0.01**: Very strong evidence. Less than 1% chance.
- **p ≥ 0.05**: No convincing evidence. Could easily be random noise. Don't deploy yet.

---

### Security & Privacy

- The module accepts user-supplied models but does not serialize, deserialize, or transmit them.
- No network calls, no data upload, no logging of customer data.
- You are responsible for the privacy and security of your own data and models.
- Random seeds are not logged or transmitted.

---

### Reproducibility & Determinism

Same data + same `random_seed` → always the same p-value (bit-for-bit identical). This is important for auditing and compliance.

---

## 4. Risks & Roadmap

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset too small or imbalanced → can't create valid test splits | Medium | High | Raise clear error if any test split has < 2 treated or < 2 control. Recommend minimum n=40 with balanced arms. |
| Slow performance on large datasets (n > 5000, many features) | Medium | Low | Document recommended `num_splits` values. Parallelization is v1.1. |
| Random assignment was not truly random in user's data | High | High | Require users to certify that pricing arm assignment is random. Document this prominently. No automatic detection. |
| Implementation differs from paper → results don't match published benchmarks | Medium | High | Validate against SEAQUAMAT data (paper's benchmark dataset). Provide replication notebook. |

---

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Analysts distrust p-values / need more context | High | Medium | Provide clear, non-technical interpretation guidance. Use language like "strong evidence" vs. "no evidence," not "significant." |
| Over-adoption without understanding assumptions (e.g., non-random arms) | Medium | Medium | Docstrings must front-load the randomization requirement. Raise an error if obvious problems detected. |

---

### Phased Rollout

#### **v1.0 (This PRD)**

**Public API:**
- `samesame.subgroup.test_segments()` — validate price-sensitive segments
- `samesame.subgroup.test_model_improvement()` — compare personalized vs. baseline pricing

**What's included:**
- Unit tests (≥ 90% code coverage)
- Integration tests (null calibration, power checks)
- SEAQUAMAT reproducibility notebook (not in CI, not bundled)
- Clear docstrings + examples for data scientists
- Module marked **experimental** in changelog and docstring

**What's internal only (no public API):**
- Balanced data splitting logic
- p-value aggregation internals
- Synthetic validation scripts

**Performance target:** K=200 splits on n=500 customers in ≤ 60 seconds

---

#### **v1.1 (Future)**

- Parallel execution (`n_jobs` parameter, joblib backend)
- Continuous outcomes (conversion revenue, not just 0/1 purchase)
- Configurable aggregation quantile (`alpha` parameter)
- Faster performance for large experiments

---

#### **v2.0 (Future)**

- Multi-arm pricing experiments (A vs. B vs. C)
- Survival outcomes (time-to-purchase)
- Adaptive pricing log support (with IPS weighting)
- Public API stability guarantee + promotion from "experimental"

---

## 5. Test Automation & Validation

### Unit Tests (in `tests/test_subgroup.py`)

| Test | What It Verifies | Pass Condition |
|------|-----------------|----------------|
| `test_segments_runs_on_valid_data` | Function executes without error on clean binary RCT data | Returns a Result with `.pvalue` |
| `test_segments_reproducible` | Same seed → identical p-value | Bit-for-bit match |
| `test_segments_rejects_non_binary_y` | Non-binary outcome raises ValueError | Error message is informative |
| `test_segments_rejects_non_binary_arm` | Non-binary pricing arm raises ValueError | Error message is informative |
| `test_model_improvement_runs` | Works with two different sklearn models | Returns a Result with `.pvalue` |
| `test_model_improvement_reproducible` | Same seed → identical p-value | Bit-for-bit match |
| `test_null_calibration` | 100 null datasets → p-values are not concentrated near 0 | Max proportion p < 0.05 is ≤ 10% |

### Integration Tests (Reproducibility)

| Test | Data | Target | Pass Condition |
|------|------|--------|----------------|
| **SEAQUAMAT replication** | Watson & Holmes (2020) benchmark dataset | Crossover test: p ≥ 0.05 (non-significant); Non-crossover: p < 0.05 (significant) | Results match published paper |

### Regression Tests

- All existing `test_api.py`, `test_iw.py`, `test_bayes.py`, `test_ood.py` pass — confirmed by running full test suite post-implementation.

---

## 6. Acceptance Criteria

| ID | What We're Testing | Pass Condition |
|----|-------------------|----------------|
| **AC-01** | API exists and is callable | `test_segments()` and `test_model_improvement()` run without error on valid data |
| **AC-02** | p-value is reproducible | Same data + same random seed → identical result (bit-for-bit) |
| **AC-03** | Binary validation | Non-binary y or pricing_arm → clear ValueError |
| **AC-04** | Model flexibility | Works with any sklearn-compatible model (LogisticRegression, RandomForest, XGBoost, etc.) |
| **AC-05** | False positive control | On 100 null datasets, ≤ 5% of p-values are < 0.05 (prevents false alarms) |
| **AC-06** | Power (segment detection) | On 100 datasets with a planted real segment, ≥ 80% detect it (p < 0.05) |
| **AC-07** | Power (model comparison) | On 100 datasets where ML model beats baseline, ≥ 70% detect it (p < 0.05) |
| **AC-08** | No regressions | All existing samesame tests pass |
| **AC-09** | Performance | K=200, n=500, RandomForest(100 trees) completes in ≤ 60 seconds |
| **AC-10** | Documentation | Docstrings explain p-values in plain language; no jargon (no "crossover TEH," "Meinshausen," etc.) |

---

## Appendix: Reference & Terminology

### Key Papers

- **Watson & Holmes (2020)**: Machine learning analysis plans for randomised controlled trials. Trials, 21(1). [Link](https://doi.org/10.1186/s13063-020-4076-y)
  - Our implementation follows this paper's framework for detecting treatment effect heterogeneity with type I error control.

---

### Plain-Language Definitions

| Term | What It Means | Why It Matters |
|------|-------------|----------------|
| **p-value** | Probability this result happened by random chance | p < 0.05 = strong evidence (not luck). p ≥ 0.05 = could be luck. |
| **Random assignment** | Each customer has 50/50 chance of seeing either price | Required for this test to work. If not true, results are meaningless. |
| **False positive** | Thinking you found a real pattern when it was just noise | Wastes money deploying strategies that don't work. |
| **Real segment** | A genuine group of customers with different price preferences | Worth deploying a personalized strategy for. |
| **Type I error** | Concluding something works when it actually doesn't | This module controls for this mathematically (p < 0.05 means ≤ 5% risk). |
| **Statistical validation** | Formal proof that a finding is real, not just luck | Replaces "it looks good in the data." |
