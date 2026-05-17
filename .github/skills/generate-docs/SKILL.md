---
name: generate-docs
description: Create or update README and MkDocs content for user-facing changes in this repository.
---

# Generate Docs

Create or update documentation that explains the library's public behavior in a
clear, beginner-friendly way.

Ask for the target audience, destination page, and behavior being documented if
not provided.

## Requirements

- Update `README.md` for landing-page or onboarding changes because
  `docs/index.md` is derived from it.
- Keep language beginner-friendly and explain statistical terms on first use.
- Use synthetic data and examples that stay aligned with the current API.
- Place content in the existing docs structure and do not hand-edit generated
  site output.
- Update examples, tutorial wording, and contributor guidance in the same
  change when they drift.