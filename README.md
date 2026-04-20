<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> Same, same but different ...

`samesame` helps you compare a reference dataset with a new one.

It answers two practical questions:

- Did anything change? Use `test_shift(...)`.
- Did things get worse? Use `test_adverse_shift(...)`.

Use it for model monitoring, data validation, drift checks, or any workflow where you need to compare two groups and decide whether the difference matters.

## Who is this for?

`samesame` is useful whenever you need to compare a reference group and a new group, for example:

- **Model monitoring** — Does production data still look like training data?
- **Data validation** — Does this new batch look like the data I expect?
- **Drift detection** — Did something change between last month and this month?
- **Group comparison** — Do two customer groups, regions, or experiments look meaningfully different?

## Installation

```bash
python -m pip install samesame
```

## Quick Start

Suppose you already have one number per row for a reference batch and a new batch.
Larger numbers mean "more concerning". That number might come from predicted risk,
anomaly detection, model confidence, or another monitoring step.

```python
import numpy as np
from samesame import test_adverse_shift, test_shift

rng = np.random.default_rng(123_456)
reference_values = rng.normal(size=600)
candidate_values = rng.normal(size=600)

shift = test_shift(reference=reference_values, candidate=candidate_values)
print(f"Did anything change? p-value = {shift.pvalue:.4f}")

harm = test_adverse_shift(
	reference=reference_values,
	candidate=candidate_values,
	direction="higher-is-worse",
)
print(f"Did things get worse? p-value = {harm.pvalue:.4f}")
```

**How to read this:** a small p-value from `test_shift(...)` suggests the new batch looks different.
A small p-value from `test_adverse_shift(...)` suggests it also looks worse.
If the first is small and the second is large, the data changed but not in a clearly harmful way.

## How it works in plain language

`samesame` does not compare raw tables directly. The usual workflow is:

1. Turn each row into one monitoring number.
2. Compare those numbers between the reference and candidate groups.

If your data has many columns, that number usually comes from a model:
predicted risk, anomaly level, model confidence, or the output of a classifier trained to tell the two groups apart.
You can think of it as a risk-like or confidence-like value for each row.

So under the hood, the package turns a multivariate dataset into one number per row, then runs two checks:
`test_shift(...)` asks whether the groups differ overall, and `test_adverse_shift(...)` asks whether the candidate group contains more of the bad end of the distribution.

## What you get back

| Function | Result type | Fields |
|----------|-------------|--------|
| `test_shift` | `ShiftResult` | `.statistic`, `.pvalue`, `.statistic_name` |
| `test_adverse_shift` | `AdverseShiftResult` | `.statistic`, `.pvalue`, `.direction` |

If you need the raw resampling values or optional Bayesian output, use `samesame.advanced`.

## Where to go next

Step-by-step examples are available in the [documentation](https://vathymut.github.io/samesame/):

**Tutorials**

- [Detect a distribution shift](https://vathymut.github.io/samesame/examples/distribution-shifts/)
- [Check whether a shift is harmful](https://vathymut.github.io/samesame/examples/noninferiority/)

**How-to guides**

- [Monitor a credit risk model](https://vathymut.github.io/samesame/examples/credit-example/)
- [Monitor model confidence](https://vathymut.github.io/samesame/examples/credit-ood-detection/)

## API at a glance

- `test_shift(*, reference, candidate, statistic="roc_auc")`
- `test_adverse_shift(*, reference, candidate, direction=...)`
- `samesame.advanced` for sample weights, more resamples, and optional Bayesian output
- `samesame.logit_scores` for turning classifier outputs into one confidence number per row
- `samesame.importance_weights` for adjusting for known group differences
- `samesame.bayes_factors` for converting between p-values and Bayes factors

Most users can keep the default settings. If your inputs are already binary 0/1 values,
`test_shift(...)` also supports `balanced_accuracy` and `matthews_corrcoef`.

## Dependencies

`samesame` has minimal dependencies. It is built on top of, and fully compatible with,
[scikit-learn][scikit-learn] and [numpy][numpy].

The public API is task-first, but it is built on standard methods for comparing per-row
monitoring values and checking whether a shift is harmful.

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
