# Unify Harmful shift evidence at the shift seam

`shift.detect_harm(...)` remains the single public seam for **Harmful shift** and always returns one `HarmResult`. Posterior draws and Bayes factor become optional top-level fields on that result, enabled with `include_posterior=False|True`, while `shift.infer_harm(...)`, `HarmInference`, the public `stats` module, and standalone Bayes conversion helpers are removed because they split one concept across shallow modules without adding leverage or locality.

The unified seam keeps one shared `n_resamples` parameter, keeps `batch` public for the permutation path only, and keeps `threshold` public but only relevant when posterior evidence is requested. Passing `threshold` while `include_posterior=False` is an error rather than a silently ignored input. When `include_posterior=False`, `HarmResult.posterior` and `HarmResult.bayes_factor` are `None` rather than sentinel values. The standard Harmful shift fields (`.statistic`, `.pvalue`, `.direction`, `.null_distribution`) must be reproducible and unchanged when `include_posterior=True` is toggled on with the same inputs and `random_state`.

Tests move to the `shift.detect_harm(...)` interface as the primary test surface. Public-style helper tests for Bayes conversion functions are removed with the deleted seams rather than preserved as a second contract.

`HarmResult` continues to extend `TestResult`, keeps `.null_distribution` mandatory, and gains optional top-level `.posterior` and `.bayes_factor` fields rather than introducing a nested evidence object or a new standalone result hierarchy. Keeping both posterior draws and the computed Bayes factor on the same result preserves leverage after the standalone helper seams are removed. When `include_posterior=True`, `shift.detect_harm(...)` is atomic: it either returns the enriched `HarmResult` or fails, rather than returning a partial result with missing posterior evidence, and the enriched result always includes both posterior draws and the Bayes-factor summary. Bayesian evidence remains specific to `shift.detect_harm(...)` and is not generalized to `shift.detect_shift(...)` in this change.

Docs follow the same seam. Advanced Harmful shift evidence moves into the main Harmful shift documentation as an advanced subsection, and the separate advanced API page is removed rather than preserved as a second documentation surface.

The posterior implementation remains a private detail under the owning Harmful shift seam rather than becoming a new named internal module with its own visible identity.
