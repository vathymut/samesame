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

`samesame` tests one score per observation — predicted risk, error, or outlier score — for change between **source** (reference) and **target** (evaluation).

- `ss.test_shift` — did anything change? Two-sided, any difference.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — did it move toward worse outcomes? One-sided, needs a direction.

## Quick example

```python
--8<-- "snippets/quick-example.py:quick-example"
```

--8<-- "snippets/pvalue-guidance.txt"

--8<-- "snippets/honest-scores.txt"

> **Toy vs real scores:** the snippet uses synthetic normals. With real data, build a score with a domain classifier — see [Detect any shift](examples/tutorials/detect-distribution-shift.md).

## Workflow

1. **Build a score** — one number per row, for source and target.
2. **Test any change** — `ss.test_shift`.
3. **Test harm** — `ss.test_harmful_shift(..., worse=...)` when you can declare the harmful direction.
4. **Weight to common support** — `ss.domain_weights` if source and target cover different regions.

Scores and weights stay fixed — only labels are permuted.

## Which test, which weights?

| Situation | What to run |
|-----------|-------------|
| Direction unknown | `test_shift` (no `worse`) |
| Direction known | `test_harmful_shift(..., worse=...)` |
| No overlap concern | omit `weights` |
| Source has outliers | `reweight="source"` |
| Target has outliers | `reweight="target"` |
| Both have outliers | `reweight="both"` |

Strings and enums are interchangeable (`"higher"` ↔ `ss.Worse.HIGHER`, `"source"` ↔ `ss.ReweightMode.SOURCE`). Weighted results are trustworthy only when [effective sample size](explanation/glossary.md#effective-sample-size-ess) stays healthy:

--8<-- "snippets/ess-rule.txt"

## Where to go next

**Tutorials** — start here (10 min):

- [1. Detect any shift](examples/tutorials/detect-distribution-shift.md) — build a score, run `test_shift`.
- [2. Is it harmful?](examples/tutorials/check-shift-harm.md) — add `worse` and interpret direction.

**How-to guides** — jump to your task:

- [Monitor credit risk, confidence, and errors](examples/credit/monitor-credit.md) — one page, three signals on HELOC.
- [Weight for common support](examples/weighting/weight-for-common-support.md) — synthetic check + HELOC, `reweight`/`shrinkage` and ESS.

Details: [When weights help](explanation/importance-weights-rationale.md) · [Why harm ≠ AUC](explanation/harmful-shift-statistic.md) · [Glossary](explanation/glossary.md) · [API](api/testing.md)

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+ (`StrEnum`), `numpy`, `scipy`, `scikit-learn`.
