<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Development Status](https://img.shields.io/badge/status-early%20development-yellow)](https://github.com/vathymut/samesame)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> Same, same but different ...

`samesame` compares a source and a target group and asks whether their score distributions differ — and whether the target moved toward worse outcomes. It is built for monitoring ML systems across time, domains, and contexts.

**Source** is the reference (usually training or a past batch); **target** is the evaluation group (usually production or a new batch). Each group is represented by a single numeric score — predicted risk, prediction error, or an outlier score such as confidence.

Two questions, two functions:

- **Did anything change?** `samesame.test_shift` — two-sided, flags any distributional difference.
- **Did it get worse?** `samesame.test_harmful_shift` — one-sided, flags a directional shift once you declare what "worse" means via `worse="higher"` or `worse="lower"`.

## Start here

- [Detect whether two datasets differ](examples/tutorials/detect-distribution-shift.md) — first end-to-end shift test.
- [Check whether target shifted toward worse outcomes](examples/tutorials/check-shift-harm.md) — when you know what "worse" means.
- [Focus on common support with importance weights](examples/tutorials/adjust-for-covariate-shift.md) — when source and target cover different feature regions.

## Quick example

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source_scores, target_scores, rng=rng)
harm = ss.test_harmful_shift(
    source_scores,
    target_scores,
    worse="higher",  # larger scores = more harm (e.g., predicted risk)
    rng=rng,
)

print(f"Shift statistic: {shift.statistic:.3f}, p-value: {shift.pvalue:.4f}")
print(f"Harm  statistic: {harm.statistic:.3f}, p-value: {harm.pvalue:.4f}")
```

A small p-value (typically ≤ 0.05) is evidence against the null. `test_shift` rejects when groups differ in any direction; `test_harmful_shift` rejects only when target carries excess mass in the harmful tail you declared.

## Choose a signal

| Signal | What it measures | `worse` |
|--------|------------------|---------|
| Predicted risk | business impact directly | `higher` |
| Prediction error (Brier, log-loss) | accuracy once labels arrive | `higher` |
| Outlier score / confidence (`LogitGap`) | certainty or anomaly level | `lower` for confidence, `higher` for anomaly |
| Domain probability `P(target\|x)` | separability of the two datasets | — (used for weighting, not as the harm signal) |

The package does not impose one signal. It gives you a small, reusable set of tests for any score.

## The workflow

1. **Build a score** for source and target.
2. **Test for any change** with `ss.test_shift`.
3. **Test for harm** with `ss.test_harmful_shift(..., worse=...)` once direction matters.

Both tests are permutation-based (default `n_resamples=9999`; use `999` while exploring). When source and target differ in feature support, `ss.domain_weights` lets you focus the comparison on [common support](explanation/importance-weights-rationale.md).

--8<-- "snippets/honest-scores.txt"

## Pick a guide

- [Monitor predicted credit risk](examples/credit/monitor-credit-risk.md) — label-free business-risk workflow.
- [Monitor model confidence](examples/credit/monitor-model-confidence.md) — when certainty matters more than the raw prediction.
- [Monitor prediction errors once labels arrive](examples/credit/monitor-prediction-errors.md) — direct accuracy monitoring.
- [Focus harmful-shift testing on common support](examples/weighting/source-reweighting.md) — down-weight source outliers outside target support.
- [Restrict testing to common support on both sides](examples/weighting/double-weighting.md) — both groups have low-overlap regions.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`.
