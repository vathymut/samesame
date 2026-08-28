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

`samesame` compares one score per observation — predicted risk, prediction error, or outlier score — between **source** (reference: training or past deployment) and **target** (current deployment).

- `ss.test_shift` — did the distribution change at all? Two-sided.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did it move toward worse outcomes? One-sided. You declare the harmful direction.

Scores and weights stay fixed; only labels are permuted to build the null.

## Quick example

```python
--8<-- "snippets/quick-example.py:quick-example"
```

Small `p` (≤ 0.05) is evidence against the null. Read `.pvalue` first, `.statistic` second — see [Get started](examples/tutorials/get-started.md) for how to interpret each.

## Workflow

1. **Build one score per observation** — source and target. If the score comes from a fitted model, generate it out of sample (cross-validation, OOB, or held-out set).
2. **Test any change** — `ss.test_shift`.
3. **Test harm** — `ss.test_harmful_shift(..., worse=...)` once you can name the harmful direction.
4. **Weight (only if needed)** — `ss.domain_weights` to focus on common support when overlap is poor.

## Where to go next

- **[Get started](examples/tutorials/get-started.md)** — 5 minutes, synthetic data, both tests end-to-end.
- **[Monitor a credit model](examples/credit/monitor-credit.md)** — real HELOC data: risk, confidence, errors.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when and how to reweight.
- **[How the harm test works](explanation/harmful-shift-statistic.md)** · **[API](api/testing.md)**

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`.
