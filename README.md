<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

**Did the target shift? Did it get worse?**

Monitoring every feature is difficult to interpret, and labels may arrive too
late to provide an early warning. `samesame` compares one meaningful score per
observation - predicted risk, prediction error, confidence, or an outlier
score - between **source** (the reference) and **target** (the current
deployment).

It answers two different questions:

- `ss.test_shift` - can the score distinguish source from target? Two-sided AUC.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` - did target move into the harmful tail you name? One-sided and tail-focused.

## Quick example

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
harm = ss.test_harmful_shift(
    source=source_scores,
    target=target_scores,
    worse="higher",  # larger = more harm (e.g., risk)
    rng=rng,
)

print(f"Shift statistic: {shift.statistic:.3f}, p-value: {shift.pvalue:.4f}")
print(f"Harm  statistic: {harm.statistic:.3f}, p-value: {harm.pvalue:.4f}")
```

Read the p-value as evidence against the relevant null, then inspect the
statistic and score distributions. A significant shift is not automatically a
harmful shift, and a p-value is not a measure of business impact.

> **Toy scores vs real scores:** synthetic normals for brevity. With real features, build a score with a domain classifier and generate it out of sample (`cross_val_predict`, `oob_decision_function_`, or held-out) — see [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md).

## Workflow

1. **One score per observation** — out of sample if from a fitted model.
2. **Any change?** `ss.test_shift`
3. **Harmful?** `ss.test_harmful_shift(..., worse=...)`
4. **Poor overlap?** `ss.domain_weights` — and only then.

Full docs: https://vathymut.github.io/samesame/

- [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md) — 5 minutes, both tests.
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit.md) — HELOC: risk, confidence, errors.
- [Weight for common support](https://vathymut.github.io/samesame/examples/weighting/weight-for-common-support.md) — when feature support differs.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`. Not for randomized experiments, subgroup discovery, or sequential alarming.
