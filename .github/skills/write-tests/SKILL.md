---
name: write-tests
description: Generate or update pytest coverage for library code, validation rules, and runnable examples.
---

# Write Tests

Generate or update pytest coverage that matches the repository's public API,
validation behavior, and documentation examples.

Ask for the target function, changed behavior, or failing scenario if not
provided.

## Requirements

- Use pytest with seeded randomness and stable synthetic data.
- Focus on public behavior, validation errors, result shapes, and documented
  examples.
- Keep tests independent, readable, and fast; use modest resample counts.
- Add regression coverage for bugs before or alongside the fix when possible.
- Prefer behavioral assertions over implementation-detail assertions.