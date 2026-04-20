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

`samesame` helps you compare a reference sample with a new sample.

It answers two practical questions:

- Did anything change? Use `test_shift(...)`.
- Did things get worse? Use `test_adverse_shift(...)`.

Use it for model monitoring, data validation, drift assessment, or any workflow where you need to compare two groups and determine whether the difference is practically important.

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

Suppose you already have one score per row for a reference sample and a new sample.
Larger scores should indicate either worse outcomes or unusual ones.
The score usually comes from a (pre-trained) model. For example, you might train a classifier to distinguish between the reference and candidate data, then use the predicted probabilities as scores. Or you might use a model's confidence or prediction errors as scores. The choice of score depends on your application and what kind of shift you want to detect.

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

**How to read this:** a small p-value from `test_shift(...)` indicates evidence that the new sample differs from the reference sample.
A small p-value from `test_adverse_shift(...)` indicates evidence that it has also shifted in a worse direction.
If the first is small and the second is large, the data changed but not in a clearly harmful way.

## How it works

`samesame` does not compare raw tables directly. The usual workflow is:

1. Turn each row into one score.
2. Compare those scores between the reference and candidate groups.

And so, the package turns a multivariate dataset into a univariate score per row, then runs two statistical tests:
`test_shift(...)` asks whether the groups differ overall, and `test_adverse_shift(...)` asks whether the candidate group is more concentrated in the adverse tail of the score distribution. Both tests are based on **permutations**, so no distributional assumptions are required.

## What you get back

| Function | Result type | Fields |
|----------|-------------|--------|
| `test_shift` | `ShiftResult` | `.statistic`, `.pvalue`, `.statistic_name` |
| `test_adverse_shift` | `AdverseShiftResult` | `.statistic`, `.pvalue`, `.direction` |

If you need the resampling distribution or optional Bayesian output, use `samesame.advanced`.

## Where to go next

Step-by-step examples are available in the [documentation](https://vathymut.github.io/samesame/):

**Tutorials**

- [Detect a distribution shift](https://vathymut.github.io/samesame/examples/tutorials/detect-distribution-shift/)
- [Check whether a shift is harmful](https://vathymut.github.io/samesame/examples/tutorials/check-shift-harm/)

**How-to guides**

- [Monitor a credit risk model](https://vathymut.github.io/samesame/examples/credit/monitor-credit-risk/)
- [Monitor prediction errors with per-sample scores](https://vathymut.github.io/samesame/examples/credit/monitor-prediction-errors/)
- [Monitor model confidence](https://vathymut.github.io/samesame/examples/credit/monitor-confidence-ood/)

## API at a glance

- `test_shift(*, reference, candidate, statistic="roc_auc")`
- `test_adverse_shift(*, reference, candidate, direction=...)`
- `samesame.advanced` for sample weights, more resamples, and optional Bayesian output
- `samesame.logit_scores` for turning classifier outputs into a confidence score
- `samesame.importance_weights` for adjusting for known group differences
- `samesame.bayes_factors` for converting between p-values and Bayes factors

Most users can keep the default settings. If your inputs are already binary 0/1 values,
`test_shift(...)` also supports `balanced_accuracy` and `matthews_corrcoef`.

## Dependencies

`samesame` has minimal dependencies. It is built on top of, and fully compatible with,
[scikit-learn][scikit-learn] and [numpy][numpy].

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
