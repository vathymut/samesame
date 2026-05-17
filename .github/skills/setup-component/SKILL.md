---
name: setup-component
description: Add or extend a module, public API entry point, or runnable example in this repository.
---

# Setup Component

Add or extend a module, public API entry point, or runnable example while
following the repository's statistical vocabulary and validation rules.

Ask for the target behavior, intended location, and whether the change is
public or private if not provided.

## Requirements

- Reuse the repository's `source` and `target` vocabulary and keep public
  parameter semantics explicit.
- Place user-facing behavior in the existing public seam when appropriate and
  keep helpers private and typed.
- Validate inputs early and keep stochastic behavior reproducible.
- Update tests and source documentation when the new component is user-visible.
- Never hand-edit generated files under `site/`.