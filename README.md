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

`samesame` compares a source group with a target group and tells you whether their score
distributions differ, and whether the target distribution shifted toward worse outcomes.

In the package, the reference group is called **source** and the new group is called **target**.
That could mean training vs production data, a baseline batch vs a fresh batch, or one segment vs
another.

The package is built around two practical questions:

- Did anything change?
- Did the change point in a worse direction?

You answer those questions with the signal that matches your use case: predicted risk, model
confidence, prediction error, or a classifier score used to compare two datasets.

## Start here

- Start with [Detect a distribution shift](examples/tutorials/detect-distribution-shift.md) if you want to know whether two datasets differ at all.
- Continue to [Check whether target shifted toward worse outcomes](examples/tutorials/check-shift-harm.md) when you know what "worse" means for your signal.
- Use [Adjust for covariate shift with importance weights](examples/tutorials/adjust-for-covariate-shift.md) when source and target have different feature coverage and you want to focus on their overlap.

## Quick example

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(123_456)
source_scores = rng.normal(size=600)
target_scores = rng.normal(size=600)

shift = ss.detect_shift(source_scores, target_scores)
harm = ss.detect_harm(
    source_scores,
    target_scores,
    direction=ss.Direction.HIGHER_IS_WORSE,
)

print(f"Shift p-value: {shift.pvalue:.4f}")
print(f"Harm  p-value: {harm.pvalue:.4f}")
```

A small p-value from `ss.detect_shift(...)` means the groups differ.
A small p-value from `ss.detect_harm(...)` means the target distribution also shifted toward worse
outcomes according to the declared direction.

## Common signals

Choose the signal that matches the decision you need to make:

- **Predicted risk** when higher values already mean higher business risk.
- **Prediction error** when labels are available and you want to measure accuracy directly.
- **Confidence score** when you want to monitor certainty rather than business impact.
- **Domain-classifier score** when your goal is to detect distribution shift between datasets.

The package does not force one interpretation on you. It gives you a small set of tests you can
reuse across these settings.

## Why it works well in practice

`samesame` is statistically grounded, but the working model is simple:

1. Build a numeric signal for source and target.
2. Test for any change with `ss.detect_shift(...)`.
3. Test for directional harm with `ss.detect_harm(...)` when direction matters.

Both tests are permutation-based, which keeps the assumptions light. When source and target differ
in feature support, `ss.from_domain_probabilities(...)` lets you focus the test on the
region where the two groups are genuinely comparable.

## Pick a guide

- [Monitor predicted credit risk](examples/credit/monitor-credit-risk.md) for a label-free business-risk workflow.
- [Monitor model confidence](examples/credit/monitor-model-confidence.md) when confidence matters more than the raw prediction.
- [Monitor prediction errors once labels arrive](examples/credit/monitor-prediction-errors.md) for direct accuracy monitoring.
- [Focus harmful-shift testing on shared support](examples/weighting/source-reweighting.md) when source contains outliers that are irrelevant for deployment.
- [Restrict testing to common support on both sides](examples/weighting/double-weighting.md) when both groups contain low-overlap outliers.

## Installation

```bash
python -m pip install samesame
```
