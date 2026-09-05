# Reviewer Rebuttal FAQ

## 1. "The contribution is incremental — D-SOS + importance weighting is a straightforward combination."

The contribution is not the combination but the estimand change. D-SOS compares observed populations; we change the comparison to the common support. This requires doubly weighted correction on both arguments, which is non-obvious because one-sided corrections (source-only or target-only) still fail under two-sided mismatch (Figure 3). The Hadamard differentiability argument is specific to Mann-Whitney-type estimands and does not follow from standard importance-weighted testing alone.

## 2. "Why not just use Crump trimming?"

Crump trimming discards observations by a hard threshold and runs an unweighted test on the remainder. It targets a different estimand and still over-rejects under two-sided mismatch (Figures 1–2). Our RIW-based approach with λ gives practitioners explicit control over the specificity-sensitivity frontier, and the doubly weighted mode corrects both sides simultaneously.

## 3. "Only 4 real-world datasets is thin."

The 4 tasks come from the TableShift benchmark and cover credit risk (HELOC), healthcare (diabetes readmission), and socioeconomic (ACS income, ACS public coverage) domains with diverse split variables. Three additional TableShift tasks are blocked by OpenML mirror issues, not by method limitations. The synthetic experiments (5 setups × 100 repeats) provide the primary statistical evidence, and the real-data results confirm the pattern across independent domains.

## 4. "The theory is standard — weighted permutation tests are well understood."

The permutation validity result (Proposition 2) shows that fixing feature-derived weights across permutations is valid because weights depend only on features, not scores — this is not the same as weighted permutation tests where weights are score-dependent. The consistency result (Proposition 1) requires a Hadamard differentiability argument specific to the Mann-Whitney functional under continuous score distributions, which goes beyond standard weighted test theory. Asymptotic properties follow from the weighted U-statistic literature (O'Neil & Redner 1993, Xie & Priebe 2002), cited in the appendix.

## 5. "The domain classifier is just logistic regression on one feature. Would it work with high-dimensional features?"

The 1D feature in synthetic experiments is a deliberate design choice to isolate overlap mechanics from domain classifier quality. The real-data tasks use the full feature space. The new domain classifier sensitivity study (Figure in appendix) shows that random forest and HistGradientBoosting produce similar calibration to logistic regression, suggesting the method is robust to classifier choice in the tested settings. Higher-dimensional settings with feature selection or regularization are a natural next step.

## 6. "How should practitioners choose λ?"

λ controls an explicit bias-variance trade-off. We recommend starting at λ = 0.5 (equal weight to density ratio and uniform weighting), consulting the ESS diagnostic, and tightening to λ = 0.25 if weak-overlap contamination is the primary concern. ESS and power trade off continuously across the λ grid; lower λ retains specificity under mismatch but concentrates weights and reduces power. Practitioners should inspect both ESS and the λ sensitivity curve to balance false-alarm protection against power loss.

## 7. "Where is this paper positioned against Bellot & van der Schaar (2021)?"

Bellot & van der Schaar establish the framework for importance-weighted two-sample testing with selection bias. Our contribution applies this framework to a one-sided harmful-shift functional (not a symmetric RKHS distance) and to overlap restriction (not confounder balance). The population target is also different: common-support comparison rather than population-level distributional equality.

## 8. "What about sequential monitoring / time series?"

The paper is scoped to non-sequential tests. Sequential risk trackers and process-control alarms introduce stopping-time and time-uniform error-control questions that require separate treatment. The common-support correction itself is modular and could be ported to sequential settings, but that is explicitly future work.
