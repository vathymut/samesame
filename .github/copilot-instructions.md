# samesame - Copilot Instructions

## Project Overview

`samesame` is a typed Python library for testing whether source and target score
distributions differ, and whether the shift points in a worse direction. The
repository combines a small public statistical API, pytest-based validation,
and MkDocs documentation aimed at practical model-monitoring use cases.

## Tech Stack

- Python 3.12+
- NumPy, SciPy, scikit-learn
- `uv` for environment and dependency management
- `pytest`, `pytest-cov`, `ruff`
- MkDocs Material, mkdocstrings, numpydoc

## Conventions

- Naming:
  - Use clear statistical names such as `source`, `target`, `direction`,
    `pvalue`, and `null_distribution` instead of generic placeholders.
  - Keep public API names stable and explicit; use leading underscores for
    private helpers.
  - Prefer descriptive constants and type aliases over abbreviations, except
    for already-established API terms such as `riw`.
- Structure:
  - Library code lives in `src/samesame/`; public seams should stay small,
    typed, and easy to discover.
  - Tests in `tests/` should mirror public API behavior and input validation.
  - Documentation lives in `README.md` and `docs/`; `docs/index.md` is derived
    from `README.md`, and `site/` is generated output that should not be
    hand-edited.
- Error handling:
  - Validate array shape, type, emptiness, finiteness, and parameter ranges
    early.
  - Raise explicit `ValueError` or `TypeError` for invalid public inputs; do
    not fail silently.
  - Keep stochastic behavior reproducible when a caller provides a random seed
    or generator.

## Workflow

- Use `uv` commands for repository tasks such as `uv sync --all-extras`,
  `uv run pytest`, `uv run ruff check .`, and `uv run mkdocs serve`.
- Prefer Conventional Commit subjects; recent history uses styles such as
  `docs:`, `refactor(api)!:`, and `chore(release):`.
- Prefer short topic branches such as `feature/...`, `fix/...`, `docs/...`, or
  `refactor/...`.
- Update tests and docs in the same change when public behavior, examples, or
  API wording changes.
- Apply the detailed guidance in these files:
  - Language guidelines: `.github/instructions/python.instructions.md`
  - Testing: `.github/instructions/testing.instructions.md`
  - Security: `.github/instructions/security.instructions.md`
  - Documentation: `.github/instructions/documentation.instructions.md`
  - Performance: `.github/instructions/performance.instructions.md`
  - Code review: `.github/instructions/code-review.instructions.md`