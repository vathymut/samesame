---
title: Context-Aware RIW Sample Weighting for Shift Testing
version: 1.1
date_created: 2026-04-20
date_updated: 2026-05-01
owner: Research & Domain Adaptation Team
tags: [spec, prd, weighting, shift-detection, domain-adaptation, covariate-shift]
---

# Specification: Context-Aware RIW Sample Weighting for Shift Testing

## 1. Executive Summary

### Problem Statement

Users of shift-detection systems need to account for known distributional differences between source (training) and target (test) datasets. Without a mechanism to reweight samples based on distributional similarity, tests may have reduced statistical power or poor calibration when group compositions differ materially.

### Solution

Context-aware importance weighting via three academically-grounded RIW strategies integrated into the `samesame` weighting API. Users pass a `ContextualRIWWeighting` strategy object through `ShiftOptions` or `AdverseShiftOptions`; no changes to function signatures are required.

### Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| SC-01 | Three weighting modes implemented and documented | 100% |
| SC-02 | Weighted shift and adverse-shift tests pass all functional and integration tests with zero regressions | 100% |
| SC-03 | `ContextualRIWWeighting` available in the public API via `ShiftOptions` / `AdverseShiftOptions` | Done |
| SC-04 | `contextual_riw()` available in `samesame.importance_weights` for standalone use | Done |

---

## 2. User Experience & Functionality

### User Personas

- **Domain Adaptation Practitioner**: Data scientist who knows source and target distributions differ and wants to improve shift-detection power.
- **Model Monitor**: ML engineer deploying production models who observes covariate shift and needs robust tests that account for known group differences.
- **Researcher**: Academic or industrial researcher studying domain adaptation and covariate shift correction.

### User Stories

**Story 1: Source-Focused Adaptation**
> Downweight training samples atypical relative to the test set, so the shift test emphasises the shared support region.

- `mode="source-reweighting"` accepted by `ContextualRIWWeighting`.
- Source samples (actual=0) receive RIW-based weights; target samples (actual=1) receive uniform weight (=1.0).

**Story 2: Test-Focused Adaptation**
> Downweight test samples atypical relative to training, so the shift test is robust to test outliers.

- `mode="target-reweighting"` accepted.
- Target samples (actual=1) receive inverse-RIW-based weights; source (actual=0) receive uniform weight (=1.0).

**Story 3: Symmetric Adaptation**
> Reweight both source and target samples to emphasise common support.

- `mode="double-weighting-covariate-shift-adaptation"` accepted.
- Source receives RIW; target receives inverse-RIW.

**Story 4: Shift Testing with Context Weights**
> Pass context membership probabilities through the standard options API.

- `ShiftOptions(weighting=ContextualRIWWeighting(...))` accepted by `test_shift()`.

**Story 5: Adverse-Shift Testing with Context Weights**
> Apply context-aware weighting to adverse-shift detection.

- `AdverseShiftOptions(weighting=ContextualRIWWeighting(...))` accepted by `test_adverse_shift()`. Both Frequentist and Bayesian modes work.

### Non-Goals

- **Automatic probability estimation**: Users must provide calibrated probabilities from a trained classifier.
- **AIW support in v1**: Only RIW-based weighting is supported. AIW is planned for v2.0.
- **Custom weighting formulas**: Only the three named modes are available.

---

## 3. Technical Scope & Assumptions

This specification covers: the three RIW-based weighting modes and their mathematical formulations; the `WeightingStrategy` API pattern; `contextual_riw()` implementation; and integration into `test_shift()` / `test_adverse_shift()` via `ShiftOptions` / `AdverseShiftOptions`.

**Audience**: Implementation engineers, test designers, and code reviewers.

**Assumptions**:
- Membership probabilities are provided by the user and are already calibrated (range (0, 1)).
- Label counts are non-zero for both groups (source and target).
- Permutation test infrastructure exists and is unchanged; weights are applied as-is to metric computations.

---

## 4. Definitions

| Term | Definition |
|------|-----------|
| **Source Group** | Samples with `actual=0`; baseline/training distribution. |
| **Target Group** | Samples with `actual=1`; new/test distribution. |
| **Membership Probability** | $\hat{p}_i = P(\text{target} \mid x_i)$; value in (0, 1). |
| **Density Ratio** | $r(x_i) = \frac{\hat{p}_i}{1 - \hat{p}_i} \cdot \frac{n_{\text{src}}}{n_{\text{tgt}}}$; prior-corrected likelihood ratio. |
| **Prior Ratio** | $\frac{n_{\text{src}}}{n_{\text{tgt}}}$; ratio of group sizes. |
| **RIW (Relative Importance Weighting)** | Weight formula: $w = \frac{r}{(1-\lambda) + \lambda \cdot r}$ for blending parameter $\lambda \in [0, 1]$. |
| **Inverse-RIW** | Algebraic inverse: $w = \frac{1}{\lambda + (1-\lambda) \cdot r}$; applied in target-reweighting mode. |
| **Context Mode** | One of three named weighting strategies: `source-reweighting`, `target-reweighting`, `double-weighting-covariate-shift-adaptation`. |
| **Common Support** | Feature space region occupied by both source and target distributions. |
| **WeightingStrategy** | Tagged union `NoWeighting \| SampleWeighting \| ContextualRIWWeighting` used by `ShiftOptions` / `AdverseShiftOptions`. |

---

## 5. Requirements, Constraints & Guidelines

### 5.1 Functional Requirements

| ID | Requirement | Details |
|----|-------------|---------|
| **REQ-001** | Three weighting modes | `source-reweighting`, `target-reweighting`, `double-weighting-covariate-shift-adaptation`. |
| **REQ-002** | RIW formula frozen | Formula: $w = \frac{r}{(1-\lambda) + \lambda \cdot r}$ with default $\lambda=0.5$. No AIW in v1. |
| **REQ-003** | Inverse-RIW formula frozen | Formula: $w = \frac{1}{\lambda + (1-\lambda) \cdot r}$; used in target-reweighting mode only. |
| **REQ-004** | Density ratio computation | Compute $r(x_i)$ from membership probabilities: $r = \frac{\hat{p}}{1-\hat{p}} \cdot \frac{n_{\text{src}}}{n_{\text{tgt}}}$. |
| **REQ-005** | Prior ratio inference | Auto-infer when `prior_ratio=None`. |
| **REQ-006** | Public API integration | `ContextualRIWWeighting` available via `samesame` public namespace; passed through `ShiftOptions` / `AdverseShiftOptions`. |
| **REQ-007** | No function-signature changes | `test_shift()` and `test_adverse_shift()` signatures are unchanged; weighting is passed via `options`. |
| **REQ-008** | Weight output range | All weights non-negative and finite; no normalization performed by `contextual_riw()`. |

### 5.2 Constraints

| ID | Constraint | Rationale |
|----|-----------|-----------|
| **CON-001** | Membership probabilities in (0, 1) | Excludes 0 and 1 to prevent division by zero in density ratio computation. |
| **CON-002** | $\lambda \in [0, 1]$ | Convex combination blending parameter; values outside this range are invalid. |
| **CON-003** | Both groups non-empty | Prior ratio inference requires $n_{\text{src}} > 0$ and $n_{\text{tgt}} > 0$. |
| **CON-004** | Binary labels | `actual` array must contain exactly values 0 (source) and 1 (target); other values invalid. |
| **CON-005** | RIW-only v1 | AIW, custom formulas, and other weighting families are out of scope. |

### 5.3 Guidelines

| ID | Guideline | Rationale |
|----|-----------|-----------|
| **GUD-001** | Validate mode string | Raise `ValueError` with all valid modes listed if mode is invalid. |
| **GUD-002** | Clear error messages | Include parameter names and valid ranges in error messages for debugging. |
| **GUD-003** | Default lam=0.5 | Default blending parameter to 0.5; document rationale (trades bias for stability). |
| **GUD-004** | Preserve backward compat | `NoWeighting` default means omitting `weighting` preserves existing behaviour. |

---

## 6. Interfaces & Data Contracts

### 6.1 `contextual_riw()` — standalone weight builder

```python
def contextual_riw(
    actual: NDArray[np.int_],
    predicted: NDArray,
    *,
    mode: ContextWeightingMode,
    lam: float = 0.5,
    prior_ratio: float | None = None,
) -> NDArray[np.float64]:
    """
    Build context-aware RIW sample weights for shift testing.
    
    Parameters
    ----------
    actual : NDArray[np.int_]
        Binary labels: 0 (source), 1 (target).
    predicted : NDArray
        Membership probabilities in (0, 1).
    mode : ContextWeightingMode
        One of: 'source-reweighting', 'target-reweighting', 
        'double-weighting-covariate-shift-adaptation'.
    lam : float, default 0.5
        RIW blending parameter in [0, 1].
    prior_ratio : float | None, default None
        Ratio n_src / n_tgt. If None, auto-inferred.
        
    Returns
    -------
    NDArray[np.float64]
        Per-sample weights, shape matching actual.length.
        
    Raises
    ------
    ValueError
        If predicted values not in (0, 1).
    ValueError
        If lam not in [0, 1].
    ValueError
        If mode is invalid.
    ValueError
        If actual not binary or has length mismatch.
    ValueError
        If prior_ratio provided but invalid.
    """
```

**Input/Output Contract**:
- Input: `actual` (length N), `predicted` (length N, values in (0, 1)), mode string.
- Output: weights (length N, dtype float64, all non-negative and finite).
- Side effects: None; function is pure.

**Raises** `ValueError` for: out-of-range probabilities; invalid `lam`; invalid `mode`; non-binary `actual`; length mismatch; empty group when `prior_ratio=None`.

### 6.2 `ContextWeightingMode` type alias

```python
ContextWeightingMode = Literal[
    "source-reweighting",
    "target-reweighting",
    "double-weighting-covariate-shift-adaptation",
]
```

- `"source-reweighting"` — RIW applied to source group (actual=0); target group weight = 1.0.
- `"target-reweighting"` — Inverse-RIW applied to target group (actual=1); source group weight = 1.0.
- `"double-weighting-covariate-shift-adaptation"` — RIW to source, inverse-RIW to target.

### 6.3 `ContextualRIWWeighting` dataclass

Defined in `samesame._weighting`; exported from `samesame` public namespace.

```python
@dataclass(frozen=True)
class ContextualRIWWeighting:
    probabilities: ArrayLike          # P(target | x) in (0, 1), pooled dataset
    mode: ContextWeightingMode        # weighting strategy
    lam: float = 0.5                  # blending parameter in [0, 1]
    prior_ratio: float | None = None  # n_src / n_tgt; inferred when None
```

### 6.4 `WeightingStrategy` tagged union

```python
WeightingStrategy = Union[NoWeighting, SampleWeighting, ContextualRIWWeighting]
```

Used by `ShiftOptions.weighting` and `AdverseShiftOptions.weighting`.

### 6.5 Integration via options objects

```python
# Shift test with context-aware weighting
result = test_shift(
    source=source_scores,
    target=target_scores,
    options=ShiftOptions(
        weighting=ContextualRIWWeighting(
            probabilities=membership_probs,
            mode="source-reweighting",
            lam=0.5,
        )
    ),
)

# Adverse-shift test with context-aware weighting
result = test_adverse_shift(
    source=source_scores,
    target=target_scores,
    direction="higher-is-worse",
    options=AdverseShiftOptions(
        weighting=ContextualRIWWeighting(
            probabilities=membership_probs,
            mode="double-weighting-covariate-shift-adaptation",
        )
    ),
)
```

### 6.6 `_resolve_weighting()` — internal resolver

```python
def _resolve_weighting(
    actual: NDArray[np.int_],
    weighting: WeightingStrategy,
) -> ArrayLike | None:
```

Called internally by `advanced.test_shift()` and `advanced.test_adverse_shift()`. Returns `None` for `NoWeighting`, explicit weights for `SampleWeighting`, or `contextual_riw(...)` output for `ContextualRIWWeighting`.

---

## 7. Acceptance Criteria

### 7.1 Functional Acceptance Criteria

| ID | Given | When | Then |
|----|-------|------|------|
| **AC-001** | `mode="source-reweighting"` | `contextual_riw()` called | Source (actual=0) have RIW weights; target (actual=1) have weight=1.0 |
| **AC-002** | `mode="target-reweighting"` | `contextual_riw()` called | Source weight=1.0; target have inverse-RIW weights |
| **AC-003** | `mode="double-weighting..."` | `contextual_riw()` called | Source get RIW; target get inverse-RIW |
| **AC-004** | `lam=0.0` | `contextual_riw()` called | Weights equal density ratio |
| **AC-005** | `lam=1.0` | `contextual_riw()` called | All weights equal 1.0 |
| **AC-006** | Invalid mode string | `contextual_riw()` called | `ValueError` with all valid modes listed |
| **AC-007** | `lam` outside [0, 1] | `contextual_riw()` called | `ValueError` |
| **AC-008** | Probabilities outside (0, 1) | `contextual_riw()` called | `ValueError` |
| **AC-009** | `ContextualRIWWeighting` in `ShiftOptions` | `test_shift()` called | Runs successfully; weights affect null distribution |
| **AC-010** | `ContextualRIWWeighting` in `AdverseShiftOptions` | `test_adverse_shift()` called | Runs successfully; Bayesian and Frequentist modes work |
| **AC-011** | Default `NoWeighting` | `test_shift()` called | Behaviour identical to pre-weighting baseline |
| **AC-012** | Default `NoWeighting` | `test_adverse_shift()` called | Behaviour identical to pre-weighting baseline |
| **AC-013** | Array length mismatch | `contextual_riw()` called | `ValueError` |
| **AC-014** | Non-binary `actual` array | `contextual_riw()` called | `ValueError` |
| **AC-015** | Empty group when `prior_ratio=None` | `contextual_riw()` called | `ValueError` |

### 7.2 Numeric Acceptance Thresholds

Test fixture: `actual = [0, 0, 1, 1]`, `predicted = [0.25, 0.4, 0.6, 0.75]`, inferred `prior_ratio=1.0`.
Density ratios: `r = [0.333, 0.667, 1.5, 3.0]`.

| ID | Mode / Condition | Expected Weights | Tolerance |
|----|-----------------|-----------------|----------|
| **NUM-001** | `source-reweighting`, lam=0.5 | `[0.5, 0.8, 1.0, 1.0]` | rtol=1e-9, atol=1e-10 |
| **NUM-002** | `target-reweighting`, lam=0.5 | `[1.0, 1.0, 0.8, 0.667]` | rtol=1e-9, atol=1e-10 |
| **NUM-003** | `double-weighting-...`, lam=0.5 | `[0.5, 0.8, 0.8, 0.667]` | rtol=1e-9, atol=1e-10 |
| **NUM-004** | Any mode, lam=1.0 | All 1.0 | rtol=1e-9, atol=1e-10 |
| **NUM-005** | Extreme r=100, lam=0.5 (RIW) | ≈1.98 (finite) | Always finite |
| **NUM-006** | Extreme r=100, lam=0.5 (Inverse-RIW) | ≈0.0198 (finite) | Always finite |
| **NUM-007** | All weights non-negative | Large array | min(weights) >= 0.0 |

---

## 8. Test Automation Strategy

### 8.1 Test Levels

**Unit Tests** (`tests/test_iw.py`): `contextual_riw()` in isolation — mode logic, boundary lam, invalid inputs, numeric values.

**Integration Tests** (`tests/test_api.py`): `test_shift()` and `test_adverse_shift()` with `ContextualRIWWeighting` — end-to-end with permutation test, backward compatibility.

**Regression Tests**: All existing tests pass unmodified (`pytest tests/`).

### 8.2 Test Fixtures

```python
actual = np.array([0, 0, 1, 1], dtype=np.int_)
predicted = np.array([0.25, 0.4, 0.6, 0.75])
# prior_ratio inferred as 1.0 (2 src, 2 tgt)
# r = [0.333, 0.667, 1.5, 3.0]
expected_source = [0.5, 0.8, 1.0, 1.0]
expected_target = [1.0, 1.0, 0.8, 0.667]
expected_double = [0.5, 0.8, 0.8, 0.667]
```

### 8.3 Key Test Cases

**`tests/test_iw.py`**

```python
def test_contextual_riw_source_mode():
    weights = contextual_riw(actual, predicted, mode="source-reweighting", lam=0.5)
    np.testing.assert_allclose(weights, expected_source, rtol=1e-9, atol=1e-10)

def test_contextual_riw_target_mode():
    weights = contextual_riw(actual, predicted, mode="target-reweighting", lam=0.5)
    np.testing.assert_allclose(weights, expected_target, rtol=1e-9, atol=1e-10)

def test_contextual_riw_double_weighting_mode():
    weights = contextual_riw(actual, predicted,
                             mode="double-weighting-covariate-shift-adaptation",
                             lam=0.5)
    np.testing.assert_allclose(weights, expected_double, rtol=1e-9, atol=1e-10)

def test_contextual_riw_invalid_mode_raises():
    with pytest.raises(ValueError, match="source-reweighting|target-reweighting"):
        contextual_riw(actual, predicted, mode="invalid-mode")

def test_contextual_riw_invalid_lam_raises():
    with pytest.raises(ValueError, match="lam must be in"):
        contextual_riw(actual, predicted, mode="source-reweighting", lam=-0.1)
```

**`tests/test_api.py`**

```python
def test_shift_with_contextual_riw():
    """ContextualRIWWeighting in ShiftOptions works end-to-end."""
    from samesame import test_shift, ShiftOptions, ContextualRIWWeighting
    probs = np.array([0.25, 0.4, 0.6, 0.75])
    result = test_shift(
        source=[0.1, 0.2, 0.3, 0.4],
        target=[0.5, 0.6, 0.7, 0.8],
        options=ShiftOptions(
            weighting=ContextualRIWWeighting(probabilities=probs,
                                             mode="source-reweighting")
        ),
    )
    assert 0.0 <= result.pvalue <= 1.0

def test_adverse_shift_with_contextual_riw():
    """ContextualRIWWeighting in AdverseShiftOptions works end-to-end."""
    from samesame import test_adverse_shift, AdverseShiftOptions, ContextualRIWWeighting
    probs = np.array([0.25, 0.4, 0.6, 0.75])
    result = test_adverse_shift(
        source=[0.1, 0.2, 0.3, 0.4],
        target=[0.45, 0.55, 0.65, 0.75],
        direction="higher-is-worse",
        options=AdverseShiftOptions(
            weighting=ContextualRIWWeighting(probabilities=probs,
                                             mode="target-reweighting")
        ),
    )
    assert 0.0 <= result.pvalue <= 1.0
```

### 8.4 CI/CD

- All tests on every commit: `pytest tests/`.
- Coverage target: >= 95% line coverage for new functions.
- Performance gate: < 5% regression vs. baseline.

---

## 9. Rationale & Context

### 9.1 Why Three Modes?

Different domain adaptation scenarios warrant different reweighting strategies:

1. **source-reweighting**: Addresses source contamination; common when source data is larger but less clean.
2. **target-reweighting**: Addresses test contamination; common when test data is small or noisy.
3. **double-weighting**: Addresses both simultaneously; most general; recommended for exploratory analysis.

### 9.2 Why RIW?

RIW is chosen over AIW for v1 because:
- Numerical stability: RIW denominator blending prevents weight explosion when density ratios are large.
- Simpler calibration: Only one blending parameter (lam) instead of two.
- Academic precedent: RIW is well-established in covariate shift adaptation literature.
- Implementation simplicity: Fewer edge cases than AIW exponentiation.

### 9.3 Why a `WeightingStrategy` Pattern?

Flat `context_*` parameters on test functions were considered but rejected in favour of the strategy pattern because:
- Avoids polluting function signatures with multiple optional parameters.
- Makes the strategy type-safe and extensible (new strategies add a new dataclass, not new params).
- Enables context-aware weighting through the **public API** without a separate "advanced API" layer.
- `ContextualRIWWeighting` is available from the `samesame` public namespace in v1.

### 9.4 Why No AIW in v1?

AIW (exponentiation-based weighting) introduces additional complexity:
- Requires tuning two parameters (exponent + lam) vs. one for RIW.
- Exponentiation can produce extreme weights when exponent is large.
- RIW achieves similar goals with simpler semantics.
- AIW support is planned for v2.0.

---

## 10. Dependencies

| Dependency | Minimum Version | Usage |
|-----------|----------------|-------|
| Python | 3.9+ | Language features |
| NumPy | 1.21+ | Array computation |
| SciPy | 1.7+ | `permutation_test` infrastructure |
| scikit-learn | — | `type_of_target()` for label validation |

---

## 11. Examples & Edge Cases

### 11.1 Standalone weight computation

```python
import numpy as np
from samesame.importance_weights import contextual_riw

actual = np.array([0, 0, 1, 1], dtype=np.int_)
predicted = np.array([0.25, 0.4, 0.6, 0.75])

weights = contextual_riw(actual, predicted, mode="source-reweighting", lam=0.5)
# [0.5, 0.8, 1.0, 1.0]
```

### 11.2 Shift test with double-weighting (public API)

```python
from samesame import test_shift, ShiftOptions, ContextualRIWWeighting
import numpy as np

result = test_shift(
    source=[0.1, 0.2, 0.3, 0.4],
    target=[0.5, 0.6, 0.7, 0.8],
    options=ShiftOptions(
        weighting=ContextualRIWWeighting(
            probabilities=np.array([0.25, 0.4, 0.6, 0.75]),
            mode="double-weighting-covariate-shift-adaptation",
            lam=0.5,
        )
    ),
)
print(result.pvalue)
```

### 11.3 Adverse-shift test (Bayesian mode)

```python
from samesame import test_adverse_shift, AdverseShiftOptions, ContextualRIWWeighting
import numpy as np

result = test_adverse_shift(
    source=[0.1, 0.2, 0.3, 0.4],
    target=[0.45, 0.55, 0.65, 0.75],
    direction="higher-is-worse",
    options=AdverseShiftOptions(
        weighting=ContextualRIWWeighting(
            probabilities=np.array([0.25, 0.4, 0.6, 0.75]),
            mode="target-reweighting",
        ),
        bayesian=True,
    ),
)
print(result.pvalue, result.bayes_factor)
```

### 11.4 Edge Case: Extreme Density Ratios

```python
# If r(x_i) >> 1.0 (target very common, source very rare):
r = 100.0
lam = 0.5

riw_weight = r / ((1 - 0.5) + 0.5 * r)  # ≈ 100 / 50.5 ≈ 1.98
inverse_riw_weight = 1.0 / (0.5 + 0.5 * r)  # ≈ 1 / 50.5 ≈ 0.0198

# Both remain finite and bounded despite large r.
# This is the numerical stability advantage of RIW over plain importance weighting.
```

### 11.5 Edge Case: lam=0 (Plain Importance Weighting)

```python
weights_riw_lam0 = contextual_riw(
    actual, predicted,
    mode="source-reweighting",
    lam=0.0
)
# All weights equal density ratio: w_i = r(x_i)
# Equivalent to plain importance weighting (AIW with exponent=1).
# May produce extreme weights; use with caution.
```

### 11.6 Edge Case: lam=1 (Uniform Weights)

```python
weights_riw_lam1 = contextual_riw(
    actual, predicted,
    mode="source-reweighting",
    lam=1.0
)
# All weights equal 1.0 (uniform).
# Equivalent to no reweighting; useful for testing that context params are passed correctly.
```

---

## 12. Validation Criteria

| Validation | Method | Gate |
|-----------|--------|------|
| **Formula Correctness** | Compare computed weights to hand-calculated values from spec | All unit tests pass |
| **API Contract** | Verify `WeightingStrategy` types accepted/rejected per spec | All integration tests pass |
| **Backward Compatibility** | Run all existing tests without modification | All regression tests pass |
| **Input Validation** | Verify all invalid inputs raise appropriate ValueError | All validation tests pass |
| **Numeric Stability** | Verify weights remain finite even with extreme density ratios | NUM-001 through NUM-007 pass |
| **Coverage** | Minimum 95% line coverage for new functions | Coverage report |
| **Performance** | Execution time not regressed vs. baseline | Performance gate < 5% slower |

---

## 13. Related Specifications / Further Reading

- **Yamada et al. (2013)**: "Relative Density-Ratio Estimation for Robust Distribution Comparison." Neural Computation, 25(5), pp. 1324-1370.
  - Foundational RIW paper; defines the mathematical framework for relative importance weighting.

- **Shimodaira (2000)**: "Improving Predictive Inference Under Covariate Shift by Weighting the Log-Likelihood Function." Journal of Statistical Planning and Inference, 90(2), pp. 227-244.
  - Early importance weighting paper; historical context for covariate shift adaptation.

- **PRD: Context-Aware RIW Weighting for Shift Testing** (`../PRD-context-aware-riw-weighting.md`)
    - _Merged into this document; PRD file deleted._

- **Source Code**: `src/samesame/importance_weights.py`, `src/samesame/_api.py`, `tests/test_iw.py`, `tests/test_api.py`
    - Implementation of all specified functions and tests.

- **`src/samesame/_weighting.py`**: `WeightingStrategy` tagged union, `ContextualRIWWeighting`, `_resolve_weighting()`.
- **`src/samesame/_types.py`**: `ShiftOptions`, `AdverseShiftOptions` with `weighting` field.
