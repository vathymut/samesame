---
applyTo: "src/**/*.py,tests/**/*.py,docs/examples/**/_code/**/*.py"
description: "Python development standards for the samesame library"
---
# Python development standards

Apply the repository-wide guidance from `../copilot-instructions.md` to all
Python changes.

## Public API

- Treat `src/samesame/shift.py` and `src/samesame/weights.py` as the public
  seam and preserve public names, parameter semantics, and result dataclasses
  unless the task explicitly changes the API.
- Keep helpers private, narrow, and colocated with the statistical behavior
  they support.
- Favor explicit return types and stable, typed result objects for user-facing
  outputs.

## Numerical and statistical code

- Accept broad array-like inputs at public boundaries, then validate and
  normalize them before computation.
- Be explicit about directionality, weighting modes, random-state handling, and
  parameter ranges; reject ambiguous states early.
- Prefer NumPy, SciPy, and scikit-learn primitives over handwritten numerical
  logic when they preserve clarity and correctness.

## Typing and readability

- Use modern Python typing, including NumPy typing, for public functions,
  dataclasses, and important intermediates.
- Prefer descriptive statistical names over abbreviations, especially for
  public arguments and result fields.
- Keep comments rare and explanatory; use docstrings to describe contracts and
  edge cases.

## Validation and reproducibility

- Route stochastic behavior through caller-provided seeds or generators and
  keep behavior deterministic when such values are supplied.
- Raise specific `ValueError` or `TypeError` exceptions for invalid public
  inputs instead of relying on downstream failures.
- Update NumPy-style docstrings when public parameters, returns, or examples
  change.