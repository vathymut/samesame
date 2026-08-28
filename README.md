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

`samesame` compares one score per observation — predicted risk, prediction error, or outlier score — between **source** (reference: training or past deployment) and **target** (current deployment).

- `ss.test_shift` — did anything change? Two-sided.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did it move toward worse outcomes? One-sided. You declare the harmful direction.

Scores and weights stay fixed; only labels are permuted.

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

Small `p` (≤ 0.05) is evidence against the null. Read `.pvalue` first, `.statistic` second.

> **Toy scores vs real scores:** the snippet uses synthetic normals for brevity. With real features, build a score with a domain classifier and go through the full loop — see [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md).

`samesame` only sees scores — not how they were made. If scores come from a fitted model, generate them out of sample with `cross_val_predict`, `oob_decision_function_`, or a held-out set.

## Workflow

1. **Build one score per observation** — source and target.
2. **Test any change** — `ss.test_shift`.
3. **Test harm** — `ss.test_harmful_shift(..., worse=...)`.
4. **Weight (only if needed)** — `ss.domain_weights` for common support.

Full docs: https://vathymut.github.io/samesame/

- [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md) — 5 minutes, both tests.
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit.md) — HELOC: risk, confidence, errors.
- [Weight for common support](https://vathymut.github.io/samesame/examples/weighting/weight-for-common-support.md) — when feature support differs.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+ (uses `StrEnum`), `numpy`, `scipy`, `scikit-learn`.
