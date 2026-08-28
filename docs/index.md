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

`samesame` tests whether a single score per observation — e.g., predicted risk, prediction error, or outlier score — has shifted between **source** (reference) and **target** (evaluation).

- `ss.test_shift` — did anything change? Two-sided.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did it move toward worse outcomes? One-sided.

## Quick example

```python
--8<-- "snippets/quick-example.py:quick-example"
```

Small `p` (≤0.05) is evidence against the null — see [Get started](examples/tutorials/get-started.md) for how to read results.

## Workflow

1. **Build a score** — a single score per observation, source and target.
2. **Test any change** — `ss.test_shift`.
3. **Test harm** — `ss.test_harmful_shift(..., worse=...)` when you can declare the direction.
4. **Weight** — `ss.domain_weights` if you need to focus on common support.

Scores and weights stay fixed — only labels are permuted.

## Where to go next

- **[Get started](examples/tutorials/get-started.md)** — 5 min: build a score, run both tests.
- **[Monitor credit](examples/credit/monitor-credit.md)** — real HELOC data: risk, confidence, and errors.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when and how to reweight.
- **[How the harm test works](explanation/harmful-shift-statistic.md)** · **[API](api/testing.md)**

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`.
