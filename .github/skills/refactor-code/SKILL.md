---
name: refactor-code
description: Refactor Python or documentation code in this repository without changing behavior unless the task explicitly requires it.
---

# Refactor Code

Refactor existing code to improve clarity, structure, or maintainability while
preserving the repository's public behavior and statistical semantics.

Ask for the target module, refactoring goal, and behavior constraints if not
provided.

## Requirements

- Preserve public API contracts unless the task explicitly requests an API
  change.
- Prefer small extractions, naming improvements, and validation consolidation
  over broad rewrites.
- Keep tests green and add coverage when the refactor reveals missing
  safeguards.
- Update docstrings or docs when the observable contract changes or becomes
  easier to understand.
- Keep changes consistent with the existing typed, minimal public seam.