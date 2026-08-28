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

`samesame` compares a source and a target group and asks whether their score distributions differ — and whether the target moved toward worse outcomes. It is built for monitoring ML systems across time, domains, and contexts.

**Source** is the reference (usually training or a past batch); **target** is the evaluation group (usually production or a new batch). Each group is represented by a single numeric score — predicted risk, prediction error, or an outlier score such as confidence.

Two questions, two functions:

- **Did anything change?** `samesame.test_shift` — two-sided, flags any distributional difference.
- **Did it get worse?** `samesame.test_harmful_shift` — one-sided, flags a directional shift once you declare what "worse" means via `worse="higher"` or `worse="lower"`.

## Quick example

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
harm = ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", rng=rng)

print(f"Shift statistic: {shift.statistic:.3f}, p-value: {shift.pvalue:.4f}")
print(f"Harm  statistic: {harm.statistic:.3f}, p-value: {harm.pvalue:.4f}")
```

A small p-value (typically ≤ 0.05) is evidence against the null. Start with `.pvalue` for evidence; use `.statistic` for magnitude. `test_shift` rejects for any difference; `test_harmful_shift` rejects only when target carries excess mass in the harmful tail you declared.

When scores come from a fitted model, generate them out of sample (cross-validation, OOB, or held-out set) — in-sample predictions invalidate the test.

## Start here

Full docs: https://vathymut.github.io/samesame/

- [Detect any distributional shift](https://vathymut.github.io/samesame/examples/tutorials/detect-distribution-shift.md)
- [Test whether the shift is harmful](https://vathymut.github.io/samesame/examples/tutorials/check-shift-harm.md)
- [Adjust for covariate shift with importance weights](https://vathymut.github.io/samesame/examples/tutorials/adjust-for-covariate-shift.md)

## Installation

```bash
python -m pip install samesame
```
