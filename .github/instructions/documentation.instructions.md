<!-- Inspired by: https://github.com/github/awesome-copilot/blob/main/instructions/update-docs-on-code-change.instructions.md -->
---
applyTo: "README.md,docs/**/*.md,CONTEXT.md,CONTRIBUTING.md"
description: "Documentation standards for samesame"
---
# Documentation standards

Apply the repository-wide guidance from `../copilot-instructions.md` whenever a
change affects user-facing behavior, examples, or contributor workflow.

## Sync requirements

- Update `README.md`, docs pages, and API wording in the same change whenever
  public behavior, installation, examples, or terminology changes.
- Treat `README.md` as the source for the docs landing page because
  `docs/index.md` is derived from it.
- Do not hand-edit `site/`; regenerate it from source documentation when
  needed.

## Structure and tone

- Keep tutorials, how-to guides, explanation pages, and API reference in the
  existing docs structure.
- Use beginner-friendly, low-jargon language and explain `source`, `target`,
  `shift`, and directional-harm terms on first use.
- Open how-to guides with the action the reader wants to take and tutorials
  with the concept the reader is learning.

## Examples

- Use synthetic, cliche, or clearly anonymized data only.
- Keep examples executable and aligned with current function names,
  parameters, and defaults.
- Prefer short examples that show the decision being made and the
  interpretation of the result.

## Completeness

- Document new parameters, default changes, return fields, and exceptions when
  public contracts change.
- Update contributor-facing docs when workflow commands, release steps, or
  documentation tooling changes.