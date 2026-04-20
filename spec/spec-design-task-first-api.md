---
title: Task-First Public API for Score-Vector Hypothesis Tests
version: 1.0
date_created: 2026-04-19
last_updated: 2026-04-19
owner: Vathy M. Kamulete
tags: [design, api, python, statistics, testing]
---

# Introduction

This specification defines the next public API for samesame. The design replaces the current class-centric, paper-language surface with a task-first, function-first API for hypothesis tests over precomputed score vectors.

## 1. Purpose & Scope

This specification covers the package's public Python API, result types, module layout, documentation focus, and migration policy.

Scope:
- Root package exports
- Primary test functions
- Advanced test functions
- Result objects
- Supporting helper namespaces
- Documentation and testing requirements for the redesign

Out of scope:
- Raw feature-matrix workflows
- Estimator training helpers
- Backward-compatibility wrappers for removed public APIs

Audience:
- Maintainers of samesame
- Contributors implementing the API redesign
- AI coding agents generating or reviewing refactor work in this repository

Assumptions:
- The library's core input is score vectors, not raw feature matrices.
- Breaking changes are acceptable for this redesign.
- Statistical rigor takes priority over configurability in the primary API.

## 2. Definitions

- **AUC**: Area Under the Receiver Operating Characteristic Curve.
- **WAUC**: Weighted Area Under the Receiver Operating Characteristic Curve.
- **CTST**: Classifier Two-Sample Test. This term remains valid in documentation context, but not as a primary public API name.
- **DSOS**: Test for harmful or adverse score shift. This term remains valid in documentation context, but not as a primary public API name.
- **Reference sample**: The baseline score vector used as the comparison anchor.
- **Candidate sample**: The new score vector being compared against the reference sample.
- **Direction**: The semantic interpretation of higher scores. Allowed values are `higher-is-worse` and `higher-is-better`.
- **Primary API**: The root-package user path intended for first use and common workflows.
- **Advanced API**: The namespace exposing expert controls, resampling details, and optional Bayesian evidence.
- **Rigorous defaults**: Defaults that prioritize statistically defensible testing behavior over speed tuning.
- **Metric registry**: An internal mapping from stable string names to direct metric callables operating on label and score arrays.

## 3. Requirements, Constraints & Guidelines

- **REQ-001**: The root package shall export exactly the primary task-oriented API needed for common score-vector workflows: `test_shift`, `test_adverse_shift`, `ShiftResult`, `AdverseShiftResult`, and `advanced`.
- **REQ-002**: The primary API shall be function-first. No public test classes shall remain in the primary or advanced API.
- **REQ-003**: The primary test functions shall require keyword-only sample arguments named `reference` and `candidate`.
- **REQ-004**: `test_shift` shall accept a named `statistic` argument with default value `roc_auc`.
- **REQ-005**: The primary API shall support only named built-in statistics. Free-form callables shall not be accepted in the primary API.
- **REQ-006**: `test_adverse_shift` shall require an explicit `direction` keyword argument.
- **REQ-007**: The primary API shall return small immutable summary result objects only.
- **REQ-008**: The primary result objects shall expose only summary fields needed for common interpretation.
- **REQ-009**: `ShiftResult` and `AdverseShiftResult` shall be distinct public result types with a shared base shape.
- **REQ-010**: The advanced API shall be exposed from `samesame.advanced` and remain function-first.
- **REQ-011**: The advanced API shall support expert controls including resampling configuration, random number generation, sample weights, and access to raw resampling artifacts.
- **REQ-012**: Bayesian evidence shall be available only in the advanced API.
- **REQ-013**: Supporting helper modules for score generation and weighting may remain in-package, but shall be demoted from the primary package story and root exports.
- **REQ-014**: The internal implementation shall use a metric registry of direct metric functions and shall not rely on `sklearn.metrics.get_scorer`.
- **REQ-015**: The documentation landing pages shall teach only the two core actions in the primary story.
- **REQ-016**: The redesign shall include tests covering the new public API and remove obsolete tests that only validate removed public classes.

- **CON-001**: The redesign is an immediate hard break. No public compatibility wrappers for `CTST`, `DSOS`, `WeightedAUC`, `nit`, or `ctst` shall be kept.
- **CON-002**: The primary API shall not expose `sample_weight`, `n_resamples`, `batch`, `rng`, `alternative`, `null_distribution`, `posterior`, or `bayes_factor`.
- **CON-003**: The primary API shall not infer score direction automatically.
- **CON-004**: The primary API shall not infer the shift statistic from input values.
- **CON-005**: Raw feature-matrix workflows are not part of this redesign and shall not be introduced as new public entry points.

- **GUD-001**: Prefer names that reflect user intent over names that reflect paper terminology.
- **GUD-002**: Keep the primary result surface small enough that a beginner can interpret it without reading implementation details.
- **GUD-003**: Keep the advanced namespace explicit enough that expert functionality is discoverable without cluttering the root package.
- **GUD-004**: Use immutable dataclasses for public result types.
- **GUD-005**: Keep supporting helper modules available, but route them through advanced documentation rather than the main landing page.

- **PAT-001**: Use one shared internal permutation-test engine for both primary and advanced APIs.
- **PAT-002**: Represent advanced detail as separate result types rather than nullable fields on primary result types.
- **PAT-003**: Validate user-facing string enums early and raise deterministic `ValueError` messages.

## 4. Interfaces & Data Contracts

### 4.1 Root Package Interface

The root package shall expose the following names:

| Name | Kind | Purpose |
| --- | --- | --- |
| `test_shift` | function | Primary test for distributional score shift |
| `test_adverse_shift` | function | Primary test for harmful score shift |
| `ShiftResult` | dataclass | Primary summary result for `test_shift` |
| `AdverseShiftResult` | dataclass | Primary summary result for `test_adverse_shift` |
| `advanced` | module | Advanced expert-oriented API namespace |

### 4.2 Primary Function Signatures

```python
from typing import Literal, TypeAlias
from numpy.typing import ArrayLike

ShiftStatistic: TypeAlias = Literal[
    "roc_auc",
    "balanced_accuracy",
    "matthews_corrcoef",
]

def test_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
) -> ShiftResult: ...

def test_adverse_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    direction: Literal["higher-is-worse", "higher-is-better"],
) -> AdverseShiftResult: ...
```

### 4.3 Primary Result Types

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float


@dataclass(frozen=True)
class ShiftResult(TestResult):
    statistic_name: str


@dataclass(frozen=True)
class AdverseShiftResult(TestResult):
    direction: Literal["higher-is-worse", "higher-is-better"]
```

The primary result types shall not expose raw resampling arrays.

### 4.4 Advanced Namespace Interface

```python
from typing import Literal
from numpy.typing import ArrayLike, NDArray

def test_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    statistic: ShiftStatistic = "roc_auc",
    n_resamples: int = 9999,
    sample_weight: ArrayLike | None = None,
    alternative: Literal["less", "greater", "two-sided"] = "two-sided",
    rng: np.random.Generator | None = None,
    batch: int | None = None,
) -> ShiftDetails: ...

def test_adverse_shift(
    *,
    reference: ArrayLike,
    candidate: ArrayLike,
    direction: Literal["higher-is-worse", "higher-is-better"],
    n_resamples: int = 9999,
    sample_weight: ArrayLike | None = None,
    bayesian: bool = False,
    rng: np.random.Generator | None = None,
    batch: int | None = None,
) -> AdverseShiftDetails: ...

@dataclass(frozen=True)
class ShiftDetails(ShiftResult):
    null_distribution: NDArray[np.float64]


@dataclass(frozen=True)
class AdverseShiftDetails(AdverseShiftResult):
    null_distribution: NDArray[np.float64]
    bayes_factor: float | None = None
    posterior: NDArray[np.float64] | None = None
```

### 4.5 Internal Data Contracts

The implementation shall support an internal metric registry:

```python
METRICS = {
    "roc_auc": roc_auc_score,
    "balanced_accuracy": balanced_accuracy_score,
    "matthews_corrcoef": matthews_corrcoef,
}
```

For adverse shift, the implementation shall use WAUC internally and flip score sign when `direction="higher-is-better"`.

### 4.6 Package Layout

The redesign shall converge on a layout similar to the following:

```text
src/samesame/
  __init__.py
  advanced.py
    bayes_factors.py
    importance_weights.py
    logit_scores.py
  _api.py
  _advanced_api.py
  _metrics.py
  _data.py
  _stats.py
  _bayesboot.py
  _utils.py
  py.typed
```

The old public modules centered on test classes shall be removed or reduced to private implementation files that are not part of the public surface.

## 5. Acceptance Criteria

- **AC-001**: Given `import samesame`, when a user inspects the root package, then the primary exported entry points are `test_shift`, `test_adverse_shift`, result types, and `advanced`.
- **AC-002**: Given a user calls `test_shift(reference=a, candidate=b)`, when both inputs are valid score vectors, then the function returns a `ShiftResult` using `roc_auc` by default.
- **AC-003**: Given a user omits `direction` for `test_adverse_shift`, when the function is called, then Python raises a missing required keyword argument error.
- **AC-004**: Given a user passes `direction="higher-is-better"`, when `test_adverse_shift` runs, then the implementation evaluates harmful shift using the sign-adjusted score orientation without requiring the user to negate inputs manually.
- **AC-005**: Given a user requests a statistic outside the built-in registry, when `test_shift` is called, then the function raises `ValueError` with the allowed statistic names.
- **AC-006**: Given a user needs `sample_weight` or resampling artifacts, when they inspect the root function signatures, then those expert controls are absent from the primary API.
- **AC-007**: Given a user imports `samesame.advanced`, when they call advanced test functions, then they can supply weights and resampling controls and receive detail-rich result types.
- **AC-008**: Given documentation landing pages are updated, when a new user reads the README or docs index, then the primary story teaches only `test_shift` and `test_adverse_shift`.
- **AC-009**: Given removed APIs such as `CTST` or `DSOS`, when a user attempts to import them from the package root, then those names are not available.
- **AC-010**: Given the refactor is complete, when the test suite runs, then tests for the new API pass and obsolete tests for removed class APIs are deleted or replaced.

## 6. Test Automation Strategy

- **Test Levels**: Unit and documentation-oriented integration tests.
- **Frameworks**: `pytest` and `pytest-cov` as already configured in this repository.
- **Test Data Management**: Keep deterministic synthetic fixtures in `tests/conftest.py`; add or adapt fixtures only when required by the new API.
- **CI/CD Integration**: Continue using the repository's pytest configuration under `pyproject.toml`.
- **Coverage Requirements**: Preserve coverage for numerical behavior, validation rules, result-shape contracts, and direction handling.
- **Performance Testing**: No dedicated performance suite is required for this redesign; performance-sensitive controls remain in the advanced namespace.

Required automated coverage areas:
- primary shift default behavior
- primary adverse-shift direction handling
- advanced weight handling
- advanced Bayesian opt-in behavior
- invalid statistic validation
- invalid direction validation
- root export surface validation

## 7. Rationale & Context

The current public API is class-based, module-heavy, and split across multiple paper-oriented names. That design leaks implementation concepts into common use and forces users to learn alternate constructors, aliases, and score-direction conventions before they can run a test.

This redesign makes the core promise explicit:
- Users bring score vectors.
- The package runs one of two hypothesis tests.
- The root package tells a two-action story.

This approach improves beginner onboarding while still preserving expert functionality in a clearly marked advanced namespace. It also reduces conceptual duplication by removing multiple public names for the same harmful-shift behavior.

## 8. Dependencies & External Integrations

### External Systems
- **EXT-001**: None. The redesign is local to the Python package API.

### Third-Party Services
- **SVC-001**: None.

### Infrastructure Dependencies
- **INF-001**: Python packaging with source layout under `src/`.

### Data Dependencies
- **DAT-001**: In-memory numeric arrays representing score vectors.

### Technology Platform Dependencies
- **PLT-001**: Python 3.12+ runtime, consistent with `pyproject.toml`.
- **PLT-002**: NumPy arrays and SciPy permutation testing.
- **PLT-003**: Direct metric functions from scikit-learn metrics.

### Compliance Dependencies
- **COM-001**: None beyond existing repository licensing and packaging constraints.

## 9. Examples & Edge Cases

```python
from samesame import test_shift, test_adverse_shift
from samesame import advanced

shift = test_shift(
    reference=train_scores,
    candidate=prod_scores,
)

harm = test_adverse_shift(
    reference=train_confidence,
    candidate=prod_confidence,
    direction="higher-is-better",
)

detail = advanced.test_shift(
    reference=train_scores,
    candidate=prod_scores,
    statistic="balanced_accuracy",
    n_resamples=4999,
    sample_weight=weights,
)
```

Edge cases:
- reference and candidate lengths may differ and shall still be supported.
- invalid statistic names shall fail fast with a clear message.
- invalid direction values shall fail fast with a clear message.
- `higher-is-better` shall internally reverse the harmful-shift score orientation.
- primary functions shall not accept positional sample arguments.
- advanced Bayesian outputs may be `None` when `bayesian=False`.

## 10. Validation Criteria

The implementation is compliant only if all of the following are true:
- root exports match the specified public surface
- primary function signatures are keyword-only and minimal
- primary results are immutable summary types without raw arrays
- advanced namespace exposes only functions, not public classes
- old class-centric public APIs are removed from the main public interface
- README and docs index focus on the two core actions
- tests are updated to the new API and pass

## 11. Related Specifications / Further Reading

- [README.md](../README.md)
- [pyproject.toml](../pyproject.toml)
- [docs/index.md](../docs/index.md)