# Claim-to-Figure Matrix

## Core claims

| Claim | Evidence target | Candidate artifact | Script | Status |
| --- | --- | --- | --- | --- |
| Unweighted harmful-shift testing can overreact when context change creates low-overlap regions | False positive rate rises under context and overlap mismatch despite no harmful change on common support | Figure 1: calibration vs overlap severity | `research/papers/dw/scripts/generate_synthetic_calibration.py` + `research/papers/dw/scripts/plot_first_figures.py` | Integrated into manuscript |
| Common-support testing improves calibration under context and overlap mismatch | Lower false positive rate than unweighted testing in null-with-mismatch settings | Table 1: null calibration summary | `research/papers/dw/scripts/generate_synthetic_calibration.py` | Integrated into manuscript |
| Common-support testing retains power for harmful change on common support | Power remains competitive when harmful shift is planted within common support | Figure 2: power vs effect size by overlap regime | `research/papers/dw/scripts/generate_power_curves.py` + `research/papers/dw/scripts/plot_first_figures.py` | Integrated into manuscript |
| Source-only, target-only, and doubly weighted tests answer different monitoring questions | Relative behavior changes under asymmetric contamination patterns | Figure 3: mode comparison under source-only, target-only, and both-side contamination | `research/papers/dw/scripts/generate_mode_comparison.py` + `research/papers/dw/scripts/plot_followup_figures.py` | Integrated into manuscript |
| The stabilization parameter controls a bias-variance trade-off | Weight dispersion, false positive rate, and power vary systematically with `lambda` | Figure 4: performance and ESS vs `lambda` | `research/papers/dw/scripts/generate_lambda_sensitivity.py` + `research/papers/dw/scripts/plot_followup_figures.py` | Integrated into manuscript |
| The method changes monitoring conclusions only where low-overlap regions would otherwise drive the result | HELOC preserves the package-doc motivating story, and the same common-support logic now appears across mirrored readmission and ACS corroboration tasks without inheriting the upstream TableShift runtime | Figure 5: HELOC spotlight plus OpenML-mirrored corroboration | `research/papers/dw/scripts/generate_real_data_workflow_summary.py` + `research/papers/dw/scripts/plot_real_data_workflow_figure.py`, backed by `research/papers/dw/scripts/real_data_workflow_sources.py` for OpenML-backed loaders that fetch pinned dataset IDs and recreate TableShift splits locally | Integrated into manuscript |

## Notes

- Keep each figure tied to a single sentence-level claim in the paper.
- Avoid figures that only show method behavior without supporting a paper claim.
- The exact script paths are recorded above, and the current result snapshot hashes are stored in the JSON metadata files under `research/papers/dw/results/`.
- Figure 5 is now backed only by the mirrored real-data workflow artifacts under `research/papers/dw/results/real_data_workflow_*`.
- Keep HELOC as the first panel or first row in the eventual real-data workflow figure so the manuscript stays aligned with the package docs.
- The old UCI readmission plus Folktables ACSEmployment corroboration path is superseded by the OpenML-backed TableShift-mirroring plan.
- The current integrated Figure 5 slate is `heloc`, `diabetes_readmission`, `acsincome`, and `acspubcov`; `physionet` is now blocked alongside `college_scorecard` and `mimic_extract_los_3` until its OpenML artifact is usable.
