---
applyTo: "tests/**/*.py,src/**/*.py,docs/examples/**/_code/**/*.py"
description: "Testing standards for samesame"
---
# Testing standards

Apply the repository-wide guidance from `../copilot-instructions.md` and
`python.instructions.md` when working on tests or testable behavior.

## Scope

- Add or update tests for every user-visible change to the public API,
  including validation errors, weighting modes, direction handling, exported
  names, and result-shape contracts.
- Prefer tests that exercise behavior through the public seam before reaching
  for private helpers.
- Keep slower or heavier statistical checks behind explicit markers or tight
  scopes.

## Determinism

- Seed random number generation in tests and keep resample counts small but
  sufficient for the behavior being asserted.
- Avoid fragile assertions on stochastic quantities unless the test controls
  the random state and tolerance carefully.
- Use fixtures and synthetic data that are stable across platforms and Python
  versions.

## Test design

- Use descriptive test names that explain the scenario and expected outcome.
- Keep tests independent, readable, and focused on one logical behavior at a
  time.
- Assert on exception type and public-facing message when validation behavior
  is part of the contract.

## Documentation and examples

- When changing runnable examples or tutorial code, update the corresponding
  tests that keep those examples honest.
- Prefer behavioral assertions over implementation details so refactors do not
  force unnecessary test rewrites.