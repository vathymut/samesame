---
description: Paper-specific terminology and framing decisions for the harmful-shift common-support paper
applyTo:
  - "research/papers/dw/**/*.tex"
  - "research/papers/dw/**/*.md"
---

# Harmful Shift Paper Memory

Locked-in decisions for terminology, framing, and conceptual distinctions in the density-weighted harmful-shift manuscript.

## Terminology Lock: Low-Overlap Observations

**Decision:** Use "observations from low-overlap regions" consistently, never "non-comparable observations" or "non-comparable cases."

**Rationale:**
- "Non-comparable" is clunky (4 syllables, Latinate prefix)
- "From low-overlap regions" is concrete and spatial (visualizable)
- Echoes the setup condition "when samples have low overlap"
- Distinguishes from "outliers" or "extreme observations"

**Cascade:** Applied in abstract, intro, experiments, conclusion.

## Conceptual Distinction: Comparability vs Data Quality

**Framing:** Observations from low-overlap regions are **not outliers**.

**Explicit statement added to intro:**
> "These observations from low-overlap regions are not outliers or corrupted data—they are valid instances that happen to be rare in the other environment. The issue is comparability, not data quality."

**Why this matters:**
- Prevents conflation with outlier detection literature
- Prevents conflation with robust learning / corruption handling
- Positions the paper correctly: distributional shift monitoring, not anomaly detection

## Abstract Parallel Structure

**Locked pattern:**
- Sentence 1: "Testing for harmful shift asks whether the target got worse."
- Sentence 2: "...standard tests can reject even when the target did not get worse on the common support..."

**Why:** The repeated phrase "got worse" / "did not get worse" creates immediate vivid contrast. Reviewers see the contradiction instantly.

## Plain Language Preference: "Got Worse" Over "Worsened"

**Decision:** Use "got worse" in abstract and intro, reserve "worsened" for method/experiments if needed.

**Rationale:**
- Abstract is for broad accessibility (program committees, interdisciplinary reviewers)
- "Got worse" is direct and concrete
- CONTEXT.md preference for formal register applies to body text, not abstract

## Common Support Definition

**Locked phrasing:** "the region where both samples have substantial overlap"

**Why "substantial":** 
- Prevents weak reading ("any non-zero overlap")
- Signals meaningful density on both sides
- Matches the technical requirement (weights are stabilized, not just non-zero)

**Applied:** First mention in abstract sentence 2, intro paragraph 1, method section definition.

## Weighting Mode Naming

**Locked terminology:**
- "Standard modes" = unweighted, source-weighted, target-weighted
- "The doubly weighted test" = our method (both sides corrected simultaneously)
- "Common-support test" = alternate name for the doubly weighted test in summaries

**Never:**
- "Weighted test" alone (ambiguous which mode)
- "Our test" (too informal)
- "RIW test" (jargon not introduced in abstract)
