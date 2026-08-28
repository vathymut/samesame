<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Development Status](https://img.shields.io/badge/status-early%20development-yellow)](https://github.com/vathymut/samesame)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v2.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

**Did the target shift? Did it get worse?**

`samesame` tests one number per observation — predicted risk, prediction error, or outlier score — for change between a **source** (reference: training or past batch) and a **target** (evaluation: current batch in production).

Two questions, two functions:

- **Did anything change?** `samesame.test_shift` — two-sided, any difference.
- **Did it get worse?** `samesame.test_harmful_shift` — one-sided, needs `worse="higher"` or `worse="lower"` (or `ss.Worse.HIGHER` / `ss.Worse.LOWER`).

## Quick example

```python
--8<-- "snippets/quick-example.py:quick-example"
```

--8<-- "snippets/pvalue-guidance.txt"

`test_shift` rejects for any difference; `test_harmful_shift` rejects only for excess mass in the harmful tail you declared.

--8<-- "snippets/honest-scores.txt"

## Choose a signal

| Signal | What it measures | `worse` |
|--------|------------------|---------|
| Predicted risk | business impact | `worse="higher"` |
| Prediction error (Brier, log-loss) | accuracy once labels arrive | `worse="higher"` |
| Outlier score — confidence (`LogitGap`) | certainty / typicality | `worse="lower"` |
| Outlier score — atypicality | distance from source | `worse="higher"` |

Domain probability `P(target | x)` is not a harm signal — use it to build importance weights, not as the score you test for harm.

Any scalar score works; `samesame` provides only the test.

## The workflow

1. **Build a score** — one number per observation — for source and target.
2. **Test any change** with `ss.test_shift`.
3. **Test harm** with `ss.test_harmful_shift(..., worse=...)` when you can declare which direction is worse.

When source and target cover different feature regions, `ss.domain_weights` focuses the comparison on [common support](explanation/importance-weights-rationale.md).

??? note "Permutation details"
    Both tests permute labels, not scores.

    --8<-- "snippets/n-resamples.txt"

    See [Glossary](explanation/glossary.md#permutation-test).

## Decide what to run

**Rule:** run unweighted first. Use `test_shift` when direction is unknown; use `test_harmful_shift` when you can declare `worse`. Add weights only when you have a domain classifier and low overlap.

- **No overlap concern** → omit `weights`.
- **Source outliers** (training has cases production never sees) → `reweight="source"`.
- **Target outliers** (production has cases training never saw) → `reweight="target"`.
- **Both** → `reweight="both"` for common-support comparison.

A weighted test is trustworthy only when [effective sample size](explanation/glossary.md#effective-sample-size-ess) stays healthy:

--8<-- "snippets/ess-rule.txt"

For the full flow, see [Glossary](explanation/glossary.md) and [When importance weights help](explanation/importance-weights-rationale.md).

## Where to go next

**Tutorials** — learn the workflow end-to-end (start here if new):

- [Detect any distributional shift](examples/tutorials/detect-distribution-shift.md) — your first shift test.
- [Test whether the shift is harmful](examples/tutorials/check-shift-harm.md) — add `worse` and interpret direction.
- [Adjust for covariate shift with importance weights](examples/tutorials/adjust-for-covariate-shift.md) — when feature support differs.

**How-to guides** — solve a specific task:

*Credit monitoring:*

- [Monitor predicted credit risk](examples/credit/monitor-credit-risk.md) — label-free business-risk workflow.
- [Monitor model confidence](examples/credit/monitor-model-confidence.md) — when certainty matters.
- [Monitor prediction errors once labels arrive](examples/credit/monitor-prediction-errors.md) — direct accuracy check.

*Common support (weighting):*

- [Focus harmful-shift testing on common support](examples/weighting/source-reweighting.md) — down-weight source outliers.
- [Restrict testing to common support on both sides](examples/weighting/double-weighting.md) — both groups have low-overlap regions.
- [Diagnose weight concentration with effective sample size](examples/weighting/diagnose-weight-concentration.md) — check a weighted result is not driven by a few points.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`.
