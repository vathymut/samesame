<!-- Inspired by: https://github.com/github/awesome-copilot/blob/main/instructions/code-review-generic.instructions.md -->
---
applyTo: "**"
description: "Code review standards for samesame"
---
# Code review standards

Apply the repository-wide guidance from `../copilot-instructions.md` and the
relevant instructions in this folder when reviewing changes.

## Findings priority

- Critical: statistical correctness bugs, unsafe validation gaps, security
  issues, breaking API changes, or missing regression coverage on changed
  public behavior.
- Important: weak tests, documentation drift, risky performance regressions,
  unnecessary dependency growth, or structural changes that obscure the public
  seam.
- Suggestion: readability, naming, or maintainability improvements that do not
  change merge readiness.

## Project-specific review focus

- Verify that changes preserve the contract of the public functions, result
  dataclasses, and exported module surface unless the task explicitly changes
  them.
- Check that stochastic behavior remains reproducible when random-state inputs
  are supplied.
- Confirm that input validation, exception types, and error messages remain
  explicit and consistent for public APIs.
- Require docs updates when `README.md`, tutorials, examples, or API wording
  should change.
- Treat direct edits under `site/` as suspect unless corresponding source
  documentation was updated and the output was regenerated.

## Review style

- Cite concrete files and lines whenever possible, and explain the user-facing
  impact of each finding.
- Distinguish bugs and regressions from taste-based suggestions.
- Prefer minimal, behavior-preserving fixes over broad rewrites when the review
  goal is remediation.