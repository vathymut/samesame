# Harmful Shift on Common Support

This context is for manuscript work under `research/papers/dw/`.

For package code, tests, docs, and examples under the shipped `samesame`
surface, use the repository root `CONTEXT.md` instead.

## Scope

- This manuscript makes a narrow claim: when deployment context changes,
  harmful-shift testing should compare source and target on common support
  rather than letting low-overlap regions drive the result.
- Scope that claim to non-sequential harmful-shift tests based on user-defined,
  rank-consistent severity scores. Treat sequential risk trackers, canary tests,
  and process-control alarms as future portability targets rather than covered
  workflows.
- The contribution is an estimand change, not a new notion of harm and not a
  general experiment-validation framework.
- Keep randomized-experiment validation, subgroup discovery, and model-lift
  comparison out of `samesame`; if that line of work continues, it belongs in a
  sibling package.

## Core framing: weighting as the contribution

- The paper changes *which observations* are compared, not *how* they are
  compared. The test (Mann-Whitney permuted with fixed weights) is a
  modular slot; the contribution is the weighting scheme.
- The three RIW modes — source-weighted, target-weighted, doubly weighted —
  are best understood as one family of weights.
- **External benchmarks** are therefore *other weighting schemes*, not other
  tests. The two primary benchmarks are:
  1. **Crump trimming** (Crump et al. 2009): hard threshold, then unweighted
  2. **Overlap weights** (Li et al. 2018, 2019): continuous $p(1-p)$ weights
- Kernel tests, MMD-based tests, and general two-sample tests are not
  primary benchmarks because they answer a different question (any-change
  detection) and do not slot into the same weighting + one-sided-test
  composition. Keep their discussion in Related Work brief and positional.
- Every experiment figure that shows rejection rates must include all
  six modes: unweighted, source-weighted, target-weighted, doubly weighted,
  Crump-trimmed, and overlap-weighted. The six-mode display is the paper's
  standard empirical canvas.
- The table at `tables/weighting_benchmarks.tex` is the single reference for
  how the three weighting approaches relate. The method section and
  experiments reference it, not a duplicated prose comparison.

## Overlap-vs-stabilized contrast (key narrative device)

- The paper must explicitly address why overlap weights and stabilized RIW
  appear similar — both are continuous, bounded, downweight extremes — and
  then separate them:
  - **Overlap weights** target $p(1-p)$, the variance of the group
    indicator. The weight shape is fixed by the propensity score alone,
    peaks at $p=0.5$, and assigns zero weight at $p\in\{0,1\}$ (soft
    version of Crump's hard trim).
  - **Stabilized RIW** targets the density ratio $r = \frac{p}{1-p} \cdot
    \frac{n_s}{n_t}$, then bounds it via $r \mapsto r/((1-\lambda)+\lambda r)$.
    It never reaches zero, incorporates the prior ratio, and has separate
    forward/inverse formulas for source vs. target.
- This contrast is the paper's sharpest pedagogical device. Every section
  that discusses external benchmarks should return to it.

## Manuscript hierarchy

- Treat the doubly weighted test as the paper's main contribution, main result,
  and main empirical object.
- Treat source-weighted and target-weighted variants as asymmetric diagnostic
  variants, not coequal headline methods.
- Reserve `distinct estimands` language for Methods. Elsewhere, describe
  source-weighted and target-weighted modes as diagnostic variants.
- Keep `lambda` as the operational bias-variance note, especially in the
  conclusion.
- The paper's empirical contribution hierarchy is:
  1. Doubly weighted RIW (our method) vs. unweighted, Crump, overlap
  2. Source-weighted and target-weighted as diagnostics
  3. Lambda sensitivity and ESS as operational guidance
  Everything else (second DGP, domain classifier sensitivity) is appendix
  corroboration.

## Standard framing

- Use the introduction's tone as the manuscript style anchor: plain statement of
  the monitoring failure first, statistical vocabulary second.
- The introduction and motivating example are a single merged section. The
  HELOC deep-dive immediately follows the problem statement; NSW provides
  corroboration that the pattern is general. No separate "Motivating Example"
  section heading.
- The core failure mode is contamination from low-overlap regions when source
  and target are compared as observed, not support change by itself.
- Related Work should treat harmful-shift testing as a broader family. D-SOS is
  the closest non-sequential score-threshold engine, not the only statistical
  test for harmful shift.
- **Within Related Work, organize around weighting approaches as benchmarks.**
  Crump, overlap, and density-ratio stabilization are the three families.
  Position them as the key comparison points. Kernel tests, MMD, and
  general any-change tests are positional (they answer a different question)
  and should appear as brief catalog entries, not as co-eval alternatives.
- The main experiments claim is that the doubly weighted test moves
  harmful-shift testing to common support and removes false alarms from
  low-overlap contamination.
- The supporting experimental claims are that source-weighted and
  target-weighted modes diagnose asymmetric contamination patterns and that
  overlap correction does not hide genuinely harmful change on common support.
- **All experiments benchmark against Crump and overlap baselines.** The
  unweighted test is the naive baseline; Crump and overlap are the
  state-of-the-art weighting approaches from causal inference; the doubly
  weighted test is our method. This three-way comparison (naive / causal
  baselines / ours) is the paper's empirical narrative arc.

## Terminology

- Use `common support` consistently; do not switch to `shared support`.
- Use `harmful shift` as the paper's own formal term. Keep `adverse shift`
  only in explicit D-SOS lineage sentences, then translate back immediately.
- Prefer `testing for harmful shift` over nominal forms such as
  `harmful-shift test` or `harmful-shift alarm`, especially in the abstract and
  introduction.
- Use `non-sequential` for the contrast with sequential tests in the monitoring
  literature.
- Use `severity score` in plain prose and define it with the D-SOS bridge:
  D-SOS calls these `outlier scores`; in monitoring language, call the
  user-chosen quantity a severity score when larger values mean worse behavior
  for the application. Mention that these can be thought of as harm scores or
  degradation scores when those labels fit the domain.
- Prefer condition-first sentence structure when stating why the doubly weighted
  mode is necessary. Example: "When both sides carry low-overlap observations,
  only the doubly weighted mode resolves the full comparability problem."
- Use em-dashes for sharp contrasts rather than italics for rhetorical stress.
  Place the contrasting element at the stress position (end of sentence).
  Example: "weighting changes emphasis, not total mass."
- Use `empirical estimand` (not `operational estimand`) for the plug-in weighted quantities.
- In the abstract, prefer `weighted test` on first mention, then `common-support test` on second mention to avoid repetition.
- Keep `source` and `target` in the abstract; do not translate to `reference`,
  `deployment`, `training`, or `production`.
- In the abstract, open with a vivid failure-mode explanation before the method
  statement. Prefer concrete statements about mistaken conclusions over a
  principle-first opening.
- Delay `severity` and `severity score` until the score object is introduced in
  the paper; in the abstract, prefer plain worsening language such as `the
  target worsens` (avoid contraction-based informality such as `gets worse`).

## Conceptual figure (intro and methods)

- The intro conceptual figure (`figures/intro_reweighting_modes.pdf`) must
  expand to show all three weighting approaches as panels or as an
  overlaid weight-function plot, not just the RIW modes.
- Option A (preferred): A single multi-panel figure where each panel shows
  the *weight as a function of domain probability* $p$ for one approach,
  plus the density-over-$x$ style for RIW modes.
- Option B: Keep the three-panel density-over-$x$ figure for RIW modes
  (source, target, both) and add a second row with Crump (binary mask)
  and Overlap (quadratic weight curve) over the same feature space. Total:
  5 panels, two rows.
- The figure caption must explicitly name all three approaches and relate
  the visual pattern to the comparison in Table~\ref{tab:weighting-benchmarks}.
- The HELOC motivating bar chart stays as-is (it shows all modes including
  Crump and Overlap if data supports it; update if needed).

## Experiments section framing

- The experiments section must open by naming the three weighting approaches as
  the comparison canvas: "We compare six modes representing three weighting
  families: none (unweighted), hard threshold (Crump-trimmed), continuous
  propensity-score weighting (overlap-weighted), and stabilized density-ratio
  weighting (source-, target-, and doubly weighted)."
- Every experimental subsection should reference Table~\ref{tab:weighting-benchmarks}
  when interpreting Crump or overlap behavior.
- The mode comparison experiment (asymmetric diagnostics) is where the
  overlap-vs-stabilized contrast lands hardest: overlap weights over-reject
  under two-sided mismatch because they lack the asymmetric correction,
  while the doubly weighted test stays calibrated. This is the paper's key
  empirical finding and must be stated prominently.
- The power experiment should note that Crump and overlap track the
  unweighted test (they also target the full observed population once
  overlap is addressed), while only the doubly weighted test separates
  common-support harm from contamination-driven signal.
- The calibration table (Table~1) must include all six modes. The current
  version shows only four — add Crump and Overlap rows.

## De-emphasized content

- Kernel two-sample tests (MMD, HSIC, etc.) appear only as positional
  citations in Related Work. They are not run as baselines because they
  answer unconditional any-change detection, not one-sided harmful shift.
- Do not add MMD-weighted or KS-weighted variants as additional experiment
  conditions — the paper's contribution is in the weights, and the
  test-agnostic slotting claim (weights apply to any two-sample test)
  belongs in the prose, not in additional experiment arms.
- Sequential monitoring methods (Podkopaev-Ramdas, conformal risk
  trackers) remain positional in Related Work. They are not benchmarks
  because the paper makes a non-sequential claim.

## Narrative economy

- Every sentence should earn its place. Avoid re-explaining the same structural
  point across multiple sections. For example, the observation that unweighted
  tests compare full populations is stated once in the introduction and relied
  upon thereafter; experimental sections do not re-derive it.
- Prefer statistical verbs over colloquial alternatives:
  `worsens` not `gets worse`, `attenuates` not `loses`, `correlates` not `lines up with`.
- Prefer `does not` over `doesn't` consistently for formal academic register.
- Prefer `one checks` over `you check` for third-person formality.
- Prefer `traced to` over `came from` or `was driven by` when attributing cause.
- Use `edge case` rather than `corner case`.
- Avoid `non-trivial`; state what is specific or nonstandard instead.
- When a sentence can be clarified by splitting or by an em-dash, prefer the
  em-dash. Avoid parenthetical clutter in long sentences.
- Avoid meta-commentary such as "This section formalizes..." or "The experiments
  ask a single question." Let the content speak for itself.
- Two figures are acceptable in the merged introduction if both earn their
  place: one motivating (concrete HELOC p-values) and one conceptual
  (reweighting modes). Do not add a third.
- Proof sketches in the main text should be tight (2--3 paragraphs at most).
  Leave extended derivations to the appendix.

## Stylistic preferences (session-confirmed)

- Conclusions should open with the resolution, not a re-explanation of the
  failure mode. Readers who reach the conclusion already understand the problem.
- Avoid ending sections with summary sentences that pre-empt the conclusion.
  Let experiment sections close on their sharpest empirical sentence.
- Prefer "The solution is to change X" over "The fix is to change X" when the
  move is a deliberate reconceptualization rather than a patch.
- Contribution bullets should reflect the paper hierarchy: main method first,
  diagnostic variants as a subordinate clause — not as a co-equal bullet.
- Rhetorical questions used as payoff lines should appear once per paper.
  Do not repeat the same question across sections.
- Cut future-work sentences from the conclusion when the limitations section
  already covers the same ground with appropriate caveats.
- Two-paragraph conclusions are preferred over three when the third paragraph
  restates material already in paragraphs one or two.
- In the abstract, avoid repeating `weighted test` across consecutive
  sentences. Use `common-support test` or `this test` as the second reference.
- In the HELOC deep-dive, lead with the concrete failure (all standard modes
  reject at p=0.002) before the structural explanation (most observations fall
  outside common support). This prioritizes the empirical shock over the
  mechanical detail.

## Writing style

- Section openings name the change or claim directly without meta-commentary.
  Do not open with "This section formalizes..." or aphoristic compression.
  Preferred register: declarative, intro-style.
  Example: "The core change is to the comparison population, not to the directional test."
- Keep the Method section focused on technical details. Do not restate contribution
  framing ("The contribution is an estimand change...") inside Method; that belongs
  in the Introduction, where reviewers expect it.
- The introduction should note that the weights slot into any two-sample test that
  accepts sample weights; this generalizability point belongs in the prose before
  the contribution bullets, not as a separate bullet.
- Mechanism vocabulary (thresholds, permutation machinery, inferential engine) belongs
  in Method. Related Work and Introduction operate at the level of questions and
  populations.
- Related Work opens with a contrast sentence dismissing the broad landscape before
  naming the closest parent. Do not open with a catalog of the family.
- Do not place self-promotion or contribution-claim sentences inside Related Work
  paragraphs. Those belong in the Introduction.
- Catalog paragraphs that list defensive citations without a positioning argument are
  debt. Prefer targeted citations placed where the technical claim is made.
- Em-dash version preferred over two separate sentences when a contrast can be fused
  at the stress position.

## Section placement rules

- Do not mention D-SOS in the abstract; keep lineage in the introduction and
  related work.
- The introduction is a single merged section that opens with the problem
  statement, immediately grounds it in the HELOC example, corroborates with
  NSW, then bridges through causal inference to D-SOS and the proposed method.
  There is no standalone "Motivating Example" section.
- In Related Work, organize prior work around weighting approaches as the
  primary structure: overlap trimming/weighting (Crump, Li), density-ratio
  weighting (Bickel, Yamada), and then positional coverage of other
  monitoring literature (risk tracking, any-change tests, conditional tests).
- Keep the Cobb comparison in Related Work, not in the introduction; state the
  distinction by monitored quantity and hypothesis.
- The abstract's evidence sentence should tie the payoff to the concrete
  failure mode: prefer `reduces false alarms from low overlap`.

## Real-data workflow

- Prefer verified OpenML mirrors loaded by pinned dataset ID and recreate the
  TableShift split rule locally.
- Treat the upstream TableShift runtime as reference-only, not as the main
  manuscript workflow.
- Active tasks: `heloc` (46932), `diabetes_readmission` (46922),
  `acsincome` (43141), `acspubcov` (43140).
- Blocked: `college_scorecard`, `physionet`, `mimic_extract_los_3`.

## Experiment infrastructure decisions

- Use `typer` for all CLI argument parsing in experiment scripts; avoid `argparse`.
- Use `polars` for tabular data manipulation in experiment scripts; prefer it over `pandas` unless a dependency (e.g., OpenML `fetch_openml`) requires the pandas interface.
- Keep the core paper module seam in `scripts/` organised as focused modules: `_dgp.py` (data-generating processes), `_io.py` (file I/O and aggregation), `_repo.py` (git and hash metadata), `_plot_utils.py` (shared plot builders), and `_domain_clf.py` (domain-probability estimators).
- Each experiment lives in a single `generate_*.py` script with a typer CLI and the `ExperimentConfig` dataclass for shared defaults.
- Each figure lives in a single `plot_*.py` script with a typer CLI.
- `render_calibration_table.py` and `verify_result_metadata.py` also use typer CLIs for consistency.
- Dead experiment scripts (e.g., `generate_overlap_baseline.py`) are removed rather than left as historical artifacts.
- CSV schema contracts live in `result_schemas.py`; no schema is defined locally in a generator script.
