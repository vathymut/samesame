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

`samesame` tests whether a single score per observation — e.g., predicted risk, prediction error, or outlier score — has shifted between a **source** (reference: training or past batch) and a **target** (evaluation: current batch in production).

Two questions, two functions:

- **Did anything change?** `samesame.test_shift` — two-sided, any difference.
- **Did it get worse?** `samesame.test_harmful_shift` — one-sided, needs `worse="higher"` or `worse="lower"` (string or `ss.Worse`, interchangeable; enum gives autocomplete).

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

Small p-value (typically ≤ 0.05) is evidence against the null. Read `.pvalue` first; use `.statistic` for magnitude. `test_shift` rejects for any difference; `test_harmful_shift` rejects only for excess mass in the harmful tail you declared. On the HELOC split used throughout the guides, expect `AUC ≈ 1.0` and `harm p < 0.001` — a clear harmful shift.

> **Toy scores vs real scores:** the snippet above uses synthetic normals for brevity. With real features, build a score with a domain classifier — see Tutorials in the docs for the full loop.

`samesame` only sees scores — not how they were made. If scores come from a fitted model, generate them out of sample with `cross_val_predict`, `oob_decision_function_`, or a held-out set — in-sample predictions invalidate the test.

## Start here

Full docs: https://vathymut.github.io/samesame/

- [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md) — build a score, run both tests (5 min).
- [Monitor credit](https://vathymut.github.io/samesame/examples/credit/monitor-credit.md) — risk, confidence, and errors on HELOC.
- [Weight for common support](https://vathymut.github.io/samesame/examples/weighting/weight-for-common-support.md) — when feature support differs.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+ (uses `StrEnum`), `numpy`, `scipy`, `scikit-learn`.
