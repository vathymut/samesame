<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
<!-- badges: end -->

**Did the target shift? Did it get worse?**

One score per observation — predicted risk, prediction error, or outlier score — for **source** (reference: training or past deployment) vs **target** (current deployment).

- `ss.test_shift` — did the distribution change at all? Two-sided AUC.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did target move into the harmful tail you name? One-sided, low-FPR weighted.

```python
--8<-- "snippets/quick-example.py:quick-example"
```

Read `.pvalue` first (≤ 0.05 is evidence against the null), `.statistic` second.

## Workflow

1. **One score per observation** — generate out of sample if the score comes from a fitted model (cross-validation, OOB, or held-out).
2. **Any change?** `ss.test_shift`
3. **Harmful?** `ss.test_harmful_shift(..., worse=...)`
4. **Poor overlap?** `ss.domain_weights` — and only then.

## Where next

- **[Get started](examples/tutorials/get-started.md)** — 5 min, both tests.
- **[Monitor a credit model](examples/credit/monitor-credit.md)** — HELOC: risk, confidence, errors.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when to reweight.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`. Not for randomized experiments, subgroup discovery, or sequential alarming.
