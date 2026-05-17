<!-- Inspired by: https://github.com/github/awesome-copilot/blob/main/instructions/security-and-owasp.instructions.md -->
---
applyTo: "src/**/*.py,tests/**/*.py,pyproject.toml,.github/workflows/**/*.yml"
description: "Security guidelines for samesame development and automation"
---
# Security guidelines

Apply the repository-wide guidance from `../copilot-instructions.md` to code,
tests, and automation that can affect safety or data handling.

## Input and numerical safety

- Validate dimensionality, emptiness, finiteness, allowed ranges, and weight
  normalization before running numerical routines.
- Reject unsupported or ambiguous states explicitly rather than silently
  coercing inputs into a plausible answer.
- Treat error handling as part of the public safety contract; do not swallow
  exceptions that would hide incorrect statistical conclusions.

## Secrets and data handling

- Never commit secrets, tokens, credentials, or private datasets to the
  repository.
- Use synthetic or clearly anonymized data in documentation, tests, and
  example code.
- Keep logs and error messages free of sensitive values or local-path
  disclosures that do not help the user act.

## Dependencies and automation

- Add new dependencies only when they are necessary, actively maintained, and
  consistent with the repository's security posture.
- Keep automation least-privilege; GitHub Actions and release tooling should
  request only the permissions they need.
- Review third-party actions, scripts, and publishing steps carefully before
  introducing them to the repository.

## Execution boundaries

- Avoid shell execution, outbound network access, or filesystem mutation in
  library code unless the behavior is explicitly part of the feature and fully
  documented.
- Treat generated AI output as untrusted input when it influences code, docs,
  or automation steps.