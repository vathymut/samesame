# Paper Workspace

This directory contains the manuscript workspace for the common-support harmful-shift paper under `research/papers/dw/`.

Use the local `CONTEXT.md` in this directory for manuscript terminology,
section hierarchy, and standing framing decisions.

## Current default

The manuscript is currently scaffolded on the official AISTATS 2026 template (10 pages). If the target venue changes, keep the section files and bibliography, and swap only the template-specific surface. The `icml2026.sty` file is also present for potential resubmission.

## Current manuscript promise

The paper now makes a narrow claim on purpose: when deployment context changes,
harmful-shift testing should compare source and target on common support rather
than letting weak-overlap regions drive the result. The point of the weighting
machinery is therefore not generic robustness or benchmark breadth. It is to
ask whether the target context is worse where the two contexts are actually
comparable.

## Writing note

Use the abstract as the style reference for the rest of the manuscript.
Lead with the overlap failure mode and the common-support question.
Keep D-SOS as background rather than headline machinery.
Prefer lean, direct phrasing over method-first inventory language.

## Layout

- `main.tex`: manuscript entry point
- `references.bib`: verified citations used by the manuscript
- `sections/`: paper sections kept as separate files
- `figures/`: committed submission figures and diagrams
- `scripts/`: reproducible figure and table generation scripts
- `notes/`: planning notes, claim-to-figure matrix, and submission tracking

## Build

From this directory:

```bash
make
```

The Makefile runs the Python-based figure and table steps through `uv run`, so
there is no separate paper-only Python environment to maintain for the current
manuscript path.

On macOS, the Makefile prefers `/Library/TeX/texbin` automatically when BasicTeX or MacTeX is installed, so you do not need to restart the shell just to make `pdflatex` visible.

That runs:

1. `pdflatex main.tex`
2. `bibtex main`
3. `pdflatex main.tex`
4. `pdflatex main.tex`

To generate the first synthetic manuscript figures without compiling the paper:

```bash
make figures
```

That target regenerates the calibration, shared-support power, asymmetric
mode-comparison, and `lambda`-sensitivity figures. The matching calibration
table can be rebuilt with:

```bash
make tables
```

To regenerate the manuscript's HELOC-led mirrored real-data workflow figure:

```bash
make real-data-figure
```

That target now uses pinned OpenML dataset IDs and recreates the TableShift
source-target split logic locally, rather than downloading data through the
upstream TableShift runtime. The current Figure~5 slate uses the four verified
mirrors already integrated into the manuscript: HELOC, diabetes readmission,
ACSIncome, and ACSPublicCoverage. To probe a smaller slice first,
override the task list and spotlight explicitly:

```bash
make real-data-figure REAL_DATA_TASKS="heloc diabetes_readmission" REAL_DATA_SPOTLIGHT=heloc
```

The paper workspace no longer carries the older synthetic case-study output or
the separate TableShift-runtime scaffold path. Figure~5 now regenerates
directly from the OpenML-backed mirrored workflow above.

Some mirrored OpenML tasks are still not executable end-to-end even when the
dataset name matches a TableShift task. In this repo state, the integrated
Figure~5 path is verified for `heloc`, `diabetes_readmission`, `acsincome`, and
`acspubcov`. The current `college_scorecard` mirror does not expose the
`CCBASIC` split column needed to recreate the TableShift institution-type
split, the current `physionet` mirror fails checksum validation through
scikit-learn/OpenML, and the current `mimic_extract_los_3` mirror does not
expose the `los_3` target or `insurance` split column.

If you prefer VS Code or Cursor, use LaTeX Workshop with the local `main.tex` entrypoint.

## Workflow

- Keep manuscript text changes on `paper/*` branches.
- Use short-lived `exp/*` branches for risky simulation or figure-generation work.
- Tag the implementation commit used for each submission-quality result snapshot.
- Prefer script-generated figures over manual edits.

## Source notes

The first-pass narrative and experiment framing came from:

- `research/specs/dw-draft.md`

The bibliography in `references.bib` currently contains only citations that were programmatically verified.

## Literature triage note

The related-work section should explicitly treat harmful-shift testing as a
family of methods, not as D-SOS alone. D-SOS remains the closest non-sequential
score-threshold engine for this manuscript: it requires a user-defined severity
score whose ranking matches the application's notion of worse, not a calibrated
risk estimate. In applications, this can be thought of as a harm score or
degradation score. Risk tracking, canary and process-control alarms,
conformal martingales, recency prediction,
resampling-based concept-drift tests, and learning-based harmful covariate-shift
tests define nearby monitoring questions. The manuscript's distinct claim is
the common-support estimand change: even a good harmful-shift alarm can answer
the wrong observed-population question when low-overlap regions dominate.
