---
name: code-review
description: Review changes in this repository for correctness, security, API stability, test quality, and documentation drift.
---

# Code Review

Review code changes against the repository's statistical contracts,
documentation standards, and validation expectations.

Ask for the review scope, changed files, and desired depth if not provided.

## Requirements

- Prioritize correctness, security, API stability, missing tests, and docs
  drift over style-only feedback.
- Cite concrete files and user impact for each finding.
- Distinguish blocking findings from non-blocking suggestions.
- Check that source documentation changed when public behavior or examples
  changed.
- Treat direct `site/` edits without source-doc changes as suspicious.