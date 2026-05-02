---
title: "PRD: Importance-Weights Documentation Roadmap"
version: 1.0
date_created: 2026-05-01
owner: Docs & Research Team
type: documentation-roadmap
tags: [prd, docs, importance-weights, weighting, shift-detection, covariate-shift]
related_prd: spec/PRD-context-aware-riw-weighting.md
---

# PRD: Importance-Weights Documentation Roadmap

## 1. Executive Summary

### Problem Statement

The \`samesame.importance_weights\` module ships three academically grounded weighting functions
(\`aiw\`, \`riw\`, \`contextual_riw\`) but the public documentation treats them as a thin API stub
(\`docs/api/importance_weights.md\` currently contains one heading and one mkdocstrings directive).

Practitioners who need covariate-shift correction for their shift tests have no worked path
showing when weighting is needed, which function to use, how to obtain valid membership
probabilities, how to set \`lam\` and \`prior_ratio\`, or how to connect raw weights to a concrete
\`test_shift\` / \`test_adverse_shift\` call. The current weighting and advanced API pages reference
importance weights but do not teach them.

### Solution

Deliver a set of new and upgraded docs pages that carry a practitioner from zero to running a
correctly weighted shift test, using narrative and code recycled from existing tutorials and
how-to guides wherever possible to preserve consistency. No API or code changes are made.

### Success Criteria

| # | Criterion | Target |
|---|-----------|--------|
| SC-01 | Every new/upgraded page is correctly routed in \`mkdocs.yml\` | 100% nav coverage |
| SC-02 | All code snippets are self-contained and runnable from scratch | Zero import errors |
| SC-03 | Practitioners can complete the new tutorial without touching any other page | End-to-end success |
| SC-04 | Every parameter (\`lam\`, \`prior_ratio\`, \`mode\`) appears at least once in a worked example | 100% parameter coverage |
| SC-05 | All cross-links between new pages and existing tutorials/how-tos resolve without 404 | 0 broken links |
| SC-06 | The \`TODO\` placeholder comment in \`docs/index.md\` is resolved by M3 | Comment removed |

---

## 2. Audience

| Audience | Role | Primary need |
|----------|------|-------------|
| **Docs maintainers** | Authors reviewing and merging docs PRs | Page spec and acceptance criteria to evaluate PRs against |
| **ML practitioners** | Data scientists and model monitors using samesame for shift testing | A clear, runnable path from "I know my data has covariate shift" to "I applied importance weights correctly" |

Both audiences are served by this roadmap: maintainers use sections 3-6 as acceptance criteria;
practitioners use the published pages that result.

---

## 3. User Goal

A practitioner reading the importance-weights documentation should be able to achieve this in sequence:

1. **Decide** whether importance weighting is needed for their shift test (or whether \`NoWeighting\` is sufficient).
2. **Choose** between \`aiw\`, \`riw\`, and \`contextual_riw\` based on their stability requirements and desired weighting mode.
3. **Implement** safely: obtain valid membership probabilities, set \`lam\` appropriately, and handle \`prior_ratio\`.
4. **Verify** by connecting weights to a concrete shift or adverse-shift test call and interpreting the result alongside existing tutorial outputs.

Every page in this roadmap must advance at least one step of this goal.

---

## 4. Current State Audit

### What exists

| File | Diataxis type | Current status |
|------|--------------|----------------|
| \`docs/api/importance_weights.md\` | Reference | **Stub only** - one heading + mkdocstrings directive |
| \`docs/api/weighting.md\` | Reference | Decent decision table + 3 code examples; cross-links to importance_weights.md |
| \`docs/api/advanced.md\` | Reference | Shows \`ContextualRIWWeighting\` in context; no weighting motivation |
| \`docs/examples/tutorials/detect-distribution-shift.md\` | Tutorial | Full worked tutorial; generates membership probabilities via \`cross_val_predict\` |
| \`docs/examples/tutorials/check-shift-harm.md\` | Tutorial | Full worked tutorial; no weighting content |
| \`docs/examples/credit/monitor-credit-risk.md\` | How-to | Full how-to; uses OOB scores, no weighted shift test |
| \`docs/examples/credit/monitor-prediction-errors.md\` | How-to | Full how-to; label-based errors, no weighting |
| \`docs/examples/credit/monitor-confidence-ood.md\` | How-to | Full how-to; LogitGap confidence scores, no weighting |
| \`docs/index.md\` | Home | Contains \`TODO\` comment blocking weighting narrative addition |

### Gaps

- No tutorial teaches the importance-weights path end-to-end.
- No how-to guide shows the three \`contextual_riw\` modes in a realistic scenario.
- No explanation page discusses when RIW is preferable to AIW, or when uniform weighting is fine.
- The API reference page is a stub with zero prose guidance.
- \`docs/index.md\` has a blocking TODO for the weighting narrative.

---

## 5. Information Architecture

### Proposed nav changes in \`mkdocs.yml\`

\`\`\`yaml
nav:
  - Home: index.md
  - Tutorials:
    - Detect a distribution shift: examples/tutorials/detect-distribution-shift.md
    - Check whether a shift is harmful: examples/tutorials/check-shift-harm.md
    - Adjust for covariate shift with importance weights: examples/tutorials/adjust-for-covariate-shift.md   # NEW
  - How-to guides:
    - Monitor a credit risk model: examples/credit/monitor-credit-risk.md
    - Monitor prediction errors: examples/credit/monitor-prediction-errors.md
    - Monitor model confidence: examples/credit/monitor-confidence-ood.md
    - Use RIW for source-focused shift testing: examples/weighting/source-reweighting.md                    # NEW
    - Use double-weighting for covariate-shift adaptation: examples/weighting/double-weighting.md           # NEW
  - Explanation:                                                                                             # NEW section
    - Why importance weights stabilise shift detection: explanation/importance-weights-rationale.md         # NEW
  - API reference:
    - Testing functions: api/testing.md
    - Advanced controls: api/advanced.md
    - Weighting strategies: api/weighting.md
    - Bayes factors: api/bayes_factors.md
    - Importance weights: api/importance_weights.md
    - Logit scores: api/logit_scores.md
\`\`\`

### New files to create

| File | Diataxis type |
|------|--------------|
| \`docs/examples/tutorials/adjust-for-covariate-shift.md\` | Tutorial |
| \`docs/examples/weighting/source-reweighting.md\` | How-to |
| \`docs/examples/weighting/double-weighting.md\` | How-to |
| \`docs/explanation/importance-weights-rationale.md\` | Explanation |

### Files to upgrade (not replace)

| File | Change |
|------|--------|
| \`docs/api/importance_weights.md\` | Add prose: decision guide, when-to-use section, parameter summary table, cross-links; keep mkdocstrings directive |
| \`docs/api/weighting.md\` | Add forward cross-link to new tutorial and explanation page |
| \`docs/index.md\` | Remove TODO comment; add one-paragraph weighting narrative to "How it works" section |

---

## 6. Page Specifications

### P1 - Tutorial: Adjust for covariate shift with importance weights

**File:** \`docs/examples/tutorials/adjust-for-covariate-shift.md\`
**Diataxis type:** Tutorial
**Learning outcome:** By the end, a practitioner can obtain membership probabilities from
\`cross_val_predict\`, pass them to \`contextual_riw\`, and run a weighted shift test.
**Prerequisite pages:** detect-distribution-shift tutorial (directly referenced).

**Sections:**

1. **What covariate shift is and why it matters for shift testing** - one paragraph; recycles the
   "classifier two-sample test" framing from \`detect-distribution-shift.md\`. Does not reproduce it verbatim.
2. **What you need** - same three-bullet pattern as the detect-distribution-shift tutorial.
3. **Step 1 - Generate membership probabilities** - recycles \`cross_val_predict\` pattern from
   \`detect-distribution-shift.md\` Steps 1-2; keeps the same \`HistGradientBoostingClassifier\` and
   \`make_classification\` setup so output values match what practitioners already know.
4. **Step 2 - Compute contextual RIW weights** - introduces \`contextual_riw\` with
   \`mode="source-reweighting"\` and default \`lam=0.5\`. Explains that source samples receive
   RIW-based weights; target samples receive unit weight. Shows \`np.round\` output.
5. **Step 3 - Run a weighted shift test** - shows \`SampleWeighting\` (passing the computed
   weights into \`advanced.test_shift\`) and contrasts unweighted vs weighted p-value.
6. **Reading the results** - two-row table: unweighted interpretation vs weighted interpretation.
7. **Tips** - lam guidance (lower = more conservative; higher = more uniform); when to use
   \`prior_ratio\`; cross-link to explanation page.

**Acceptance criteria:**

- All imports resolve from \`samesame\` public API only.
- Steps 1-3 produce deterministic output when \`random_state=123_456\` is set.
- Tutorial does not explain *why* RIW is used; that belongs in the explanation page.
- Cross-links to \`detect-distribution-shift.md\` (prerequisite) and
  \`explanation/importance-weights-rationale.md\` (deepening).

---

### P2 - How-to: Use RIW for source-focused shift testing

**File:** \`docs/examples/weighting/source-reweighting.md\`
**Diataxis type:** How-to
**Problem statement:** You have a credit risk model deployed on a population that partially
overlaps with training. You want the shift test to emphasize the shared region and de-emphasize
training samples that are completely foreign to the deployment population.
**Prerequisite pages:** Tutorial P1 (adjust-for-covariate-shift).

**Narrative origin:** Recycles setup from \`monitor-credit-risk.md\` (HELOC dataset, OOB scores,
\`ExternalRiskEstimate\` split). Adds the weighting layer on top rather than replacing the
unweighted test.

**Sections:**

1. **The scenario** - one paragraph; directly references \`monitor-credit-risk.md\` as prior
   reading and states the additional weighting goal.
2. **Step 1 - Reproduce the unweighted shift test** - four-line snippet calling \`test_shift\`
   with no weighting; expected output recycled from \`monitor-credit-risk.md\`.
3. **Step 2 - Obtain membership probabilities** - shows fitting a second classifier (separate
   from the credit model) to distinguish training from deployment; uses \`cross_val_predict\`.
4. **Step 3 - Apply source-reweighting** - \`contextual_riw(..., mode="source-reweighting")\`,
   then wraps weights in \`SampleWeighting\` and passes to \`advanced.test_shift\`.
5. **Step 4 - Compare weighted vs unweighted results** - side-by-side table of statistic and
   p-value; interprets the direction of change.
6. **When to use this mode** - bullet list: common support is narrow; training has many
   out-of-distribution outliers; you want the test to focus on the overlap region.
7. **Cross-links** - to \`double-weighting.md\` (both groups reweighted), to explanation page.

**Acceptance criteria:**

- The snippet in Step 4 is self-contained when run after Steps 2-3.
- The guide does not re-explain \`riw\` vs \`contextual_riw\` internals; it cross-links to P4.
- Output values are reproducible with \`random_state=12345\`.

---

### P3 - How-to: Use double-weighting for covariate-shift adaptation

**File:** \`docs/examples/weighting/double-weighting.md\`
**Diataxis type:** How-to
**Problem statement:** You want the shift test to focus exclusively on the region of feature
space shared by both source and target, reweighting both groups symmetrically.
**Prerequisite pages:** Tutorial P1 and How-to P2 (source-reweighting as simpler alternative).

**Narrative origin:** Reuses HELOC setup from P2. Introduces \`mode="double-weighting-covariate-shift-adaptation"\`.

**Sections:**

1. **When single-group reweighting is not enough** - two-sentence motivation: if the target group
   also contains outliers foreign to the source, source-only reweighting leaves those uncorrected.
2. **Step 1 - Obtain membership probabilities** - one-liner reference to P2 Step 2 (same code, no repetition).
3. **Step 2 - Apply double-weighting** - \`contextual_riw(..., mode="double-weighting-covariate-shift-adaptation")\`.
4. **Step 3 - Run the weighted adverse-shift test** - uses \`advanced.test_adverse_shift\` to show
   the mode works with both shift functions.
5. **Step 4 - Interpret the result** - three-row comparison table: no weighting / source-reweighting / double-weighting.
6. **Choosing \`lam\`** - explains the \`lam=0.5\` default and when to raise or lower it; references
   the formula from the explanation page.
7. **Cross-links** - back to P2, forward to explanation page, to \`api/weighting.md\`.

**Acceptance criteria:**

- Step 3 uses \`advanced.test_adverse_shift\` (not \`test_shift\`) to demonstrate generality.
- The guide explicitly states that double-weighting is the most aggressive mode and should be
  chosen only when outliers are present in both groups.
- Output values reproducible with \`random_state=12345\`.

---

### P4 - Explanation: Why importance weights stabilise shift detection

**File:** \`docs/explanation/importance-weights-rationale.md\`
**Diataxis type:** Explanation
**Discussion goal:** Explain the conceptual landscape - why plain IWERM can produce extreme
weights, what AIW's \`lam\` parameter trades off, why RIW's denominator blending is more stable,
and when each of the three \`contextual_riw\` modes is appropriate.

**Sections:**

1. **The density-ratio problem** - describes the density ratio formula and why extreme values arise
   when the classifier separates groups well.
2. **AIW: taming the ratio with a power parameter** - the AIW formula; describes the \`lam=0\`
   (uniform) to \`lam=1\` (exact density ratio) spectrum; warns that \`lam=1\` is IWERM and can
   still produce extreme weights.
3. **RIW: blending the denominator** - the RIW formula; explains that the blended denominator caps
   maximum weight magnitude; describes what happens at \`lam=0\` (reduces to IWERM) and \`lam=1\` (uniform).
4. **Contextual RIW: three modes** - plain-language description of source-reweighting,
   target-reweighting, and double-weighting; uses a distribution-overlap diagram or ASCII art to
   ground the concepts; no code blocks on this page.
5. **Prior ratio** - explains the n_tr/n_te correction and when to override it with a custom value.
6. **Decision guide** - three-row table mapping scenario to function:

   | Scenario | Recommended function | Key parameter |
   |----------|---------------------|---------------|
   | First-pass stability check; moderate shift | \`riw\` or \`aiw\` with \`lam=0.5\` | \`lam\` |
   | Shift test focused on common support, source outliers only | \`contextual_riw\`, source-reweighting | \`mode\`, \`lam\` |
   | Shift test focused on common support, outliers in both groups | \`contextual_riw\`, double-weighting | \`mode\`, \`lam\` |

7. **References** - Shimodaira (2000) and Yamada et al. (2013) cited inline with the same
   format used in the module docstrings.

**Acceptance criteria:**

- No code blocks. This is a discussion page, not a recipe.
- Every formula uses MathJax (already enabled in \`mkdocs.yml\`).
- Decision guide table matches the three modes exposed in \`ContextWeightingMode\`.
- Does not describe implementation details of \`_density_ratio\` or any private functions.

---

### P5 - Upgraded API reference: Importance weights

**File:** \`docs/api/importance_weights.md\`
**Change type:** Upgrade (not replacement)
**Diataxis type:** Reference (retains mkdocstrings directive as primary content)

**Prose to add before the mkdocstrings directive:**

The upgraded page must include:

- A **when-to-use paragraph** explaining that weights are useful when covariate shift is known
  and the test should focus on common support.
- A **choosing-a-function table** with one-line rationale per function:
  - \`aiw\`: exponentially tempered density ratio; \`lam=1\` is IWERM; \`lam=0\` is uniform.
  - \`riw\`: numerically stable via blended denominator; \`lam=0.5\` is the safe default.
  - \`contextual_riw\`: reweight source only, target only, or both; use with \`ContextualRIWWeighting\`.
- A **connecting-weights-to-a-test paragraph** pointing to \`api/weighting.md\`.
- Cross-links to: the new explanation page (conceptual), the new tutorial P1 (worked example),
  and \`api/weighting.md\` (integration).

**Acceptance criteria:**

- All three functions appear in the decision table.
- mkdocstrings directive is preserved unchanged.
- All three cross-links resolve without 404.

---

### P6 - \`docs/index.md\` TODO resolution

**Change type:** Targeted edit (one paragraph added; TODO comment removed)

**What to remove:**

\`\`\`html
<!-- TODO(agents): The contextual weighting narrative belongs here once samesame.weighting
stories are fully fleshed out. Do NOT expand or add content to this section until that work
is complete and this comment is removed. -->
\`\`\`

**Replacement paragraph** (two sentences maximum, inserted directly after the permutation-based
paragraph in the "How it works" section):

> When you know that source and target have different feature distributions - covariate shift -
> you can supply per-sample importance weights to focus the test on the region where both groups
> overlap. See [Adjust for covariate shift with importance weights](/examples/tutorials/adjust-for-covariate-shift/).

**Acceptance criteria:**

- TODO comment is removed.
- Replacement is two sentences maximum.
- Paragraph links to tutorial P1.

---

## 7. Reuse Map

Content from existing pages is recycled and reframed. Prose and code blocks must not be
copied verbatim; they must be reframed to match each new page's goal.

| Existing content | Source page | Recycled in | How it is reframed |
|-----------------|-------------|-------------|-------------------|
| \`cross_val_predict\` pattern (Steps 1-2) | detect-distribution-shift tutorial | P1 Tutorial (Steps 1-2) | Same sklearn setup; adds \`contextual_riw\` after probability generation |
| HELOC dataset setup + OOB scoring | monitor-credit-risk how-to | P2 How-to (Steps 1-2) | Steps 1-2 are referenced, not repeated; focus shifts to weighting layer |
| OOB + \`predict_proba\` split | monitor-credit-risk how-to | P3 How-to (Step 1) | Condensed to one-liner reference; avoids re-introducing HELOC from scratch |
| \`ContextualRIWWeighting\` snippet | api/weighting.md + api/advanced.md | P2, P3 (Steps 3-4) | Expanded with output, interpretation table, and \`lam\` discussion |
| \`test_adverse_shift\` with \`direction\` | check-shift-harm tutorial | P3 How-to (Step 3) | Reused to demonstrate double-weighting with adverse-shift, not plain shift |

---

## 8. Migration Plan for \`docs/api/importance_weights.md\`

**Current state:** 6-line stub.
**Target state (after M2):** Full reference page with prose sections + mkdocstrings autorenderer.

| Step | Milestone | Action |
|------|-----------|--------|
| 1 | M1 | Confirm file is in nav and mkdocstrings renders without errors |
| 2 | M2 | Insert prose sections above the directive per P5 spec |
| 3 | M3 | QA: verify all cross-links resolve; verify mkdocstrings output unchanged; check against CONTEXT.md terminology |

---

## 9. Rollout Plan

### M1 - Structure (no prose content)

**Goal:** All new files exist in the nav; all paths render without 404.

| Task | Output |
|------|--------|
| Add four new stub files at target paths | Empty files with correct headings |
| Update \`mkdocs.yml\` nav per Section 5 | Nav entries in place |
| Run \`mkdocs build\` and confirm zero warnings on new paths | Build report |

**Done when:** \`mkdocs serve\` shows all new pages in nav without 404.

---

### M2 - Content

**Goal:** All new pages have full prose and runnable code; upgraded pages have full prose additions.

**Authoring order** (dependency order):

1. P4 (Explanation) - no code, no page dependencies; anchors terminology for all other pages.
2. P5 (API reference upgrade) - builds on P4 cross-links.
3. P1 (Tutorial) - depends on P4 for deepening cross-link; recycled from detect-distribution-shift.
4. P2 (How-to source-reweighting) - depends on P1 as stated prerequisite.
5. P3 (How-to double-weighting) - depends on P2 as stated prerequisite.
6. P6 (index.md TODO resolution) - can be done in parallel with P1.

**Done when:** All six pages have full content per section specs in Section 6.

---

### M3 - Polish and Validation

**Goal:** All SC-01 through SC-06 acceptance criteria pass.

| Task | Check |
|------|-------|
| Run all code snippets in a clean virtual environment | Zero import errors |
| Verify all internal cross-links | Zero 404s |
| Terminology audit against \`CONTEXT.md\` glossary | No term conflicts |
| Confirm TODO comment removed from \`docs/index.md\` | SC-06 passes |
| \`mkdocs build --strict\` | Zero warnings |

**Done when:** All five tasks pass and SC-01 to SC-06 are green.

---

## 10. Non-Goals

- No changes to \`src/samesame/importance_weights.py\` or any other source file.
- No new statistical methods; \`aiw\`, \`riw\`, and \`contextual_riw\` are the only functions documented.
- No changes to the MkDocs theme or design system beyond nav updates.
- No changes to \`spec/PRD-context-aware-riw-weighting.md\` (feature PRD stays as-is).
- \`target-reweighting\` mode gets no dedicated how-to in v1; it is covered in the explanation page and API reference.
- No changes to existing tutorial or how-to pages beyond the targeted cross-link additions in P2 and P3.

---

## 11. Cross-References

| Document | Relationship |
|----------|-------------|
| \`spec/PRD-context-aware-riw-weighting.md\` | Feature PRD this roadmap implements documentation for |
| \`CONTEXT.md\` | Canonical domain glossary; all new pages must align with its terminology |
| \`src/samesame/importance_weights.py\` | Source of truth for function signatures, defaults, and docstring examples |
| \`docs/api/weighting.md\` | Integration point; receives forward cross-links in M2 |
| \`docs/examples/tutorials/detect-distribution-shift.md\` | Primary reuse source for P1 (Tutorial) |
| \`docs/examples/credit/monitor-credit-risk.md\` | Primary reuse source for P2 and P3 (How-tos) |
