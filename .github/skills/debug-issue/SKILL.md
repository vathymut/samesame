---
name: debug-issue
description: Debug failing behavior in code, tests, or examples for this repository.
---

# Debug Issue

Debug failing behavior in the library, tests, or runnable examples using a
small-hypothesis, fast-validation workflow.

Ask for the failing command, failing test, or observed behavior if not
provided.

## Requirements

- Reproduce the issue first when possible, then form a narrow local
  hypothesis.
- Fix the root cause with the smallest safe edit.
- Validate with the cheapest focused check after each substantive change.
- Control randomness during debugging with explicit seeds or generators.
- Add regression tests and docs updates when public behavior changes.