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

`samesame` runs hypothesis tests over score vectors.

It gives you two primary actions:

- `test_shift` asks whether two score distributions differ.
- `test_adverse_shift` asks whether the new scores got worse in a direction you define explicitly.

This is useful when you already have per-row scores from a classifier, anomaly detector,
risk model, or confidence heuristic and want statistically defensible answers instead of ad hoc thresholds.

## Who is this for?

`samesame` is useful whenever you need to compare two datasets statistically, for example:

- **Model monitoring** — Is my production data starting to look different from my training data?
- **Data validation** — Does this new data batch match the distribution I expect?
- **Drift detection** — Has the input distribution shifted between last month and this month?
- **A/B testing** — Are the two groups I'm comparing actually comparable?

## Installation

```bash
python -m pip install samesame
```

## Quick Start

The example below shows why having *two* tests matters.

Imagine you have outlier scores from a training set and a test set — higher score means more unusual.
You want to know: (a) are the distributions different? and (b) is the test set actually *worse*?

```python
import numpy as np
from samesame import test_adverse_shift, test_shift

rng = np.random.default_rng(123_456)
os_train = rng.normal(size=600)  # outlier scores from training
os_test  = rng.normal(size=600)  # outlier scores from deployment

# Question 1: Are the distributions different?
shift = test_shift(reference=os_train, candidate=os_test)
print(f"Shift p-value: {shift.pvalue:.4f}")

# Question 2: Is the test set actually worse (more outliers)?
harm = test_adverse_shift(
	reference=os_train,
	candidate=os_test,
	direction="higher-is-worse",
)
print(f"Adverse-shift p-value: {harm.pvalue:.4f}")
```

**What this means:** `test_shift` flags a difference (the distributions are not identical), but `test_adverse_shift` says
the test set is *not* disproportionately worse. This is a common real-world situation — minor
statistical differences that do not signal a real problem. Without a harmful-shift test, you might raise a false alarm.

## What you get back

| Function | Result type | Fields |
|----------|-------------|--------|
| `test_shift` | `ShiftResult` | `.statistic`, `.pvalue`, `.statistic_name` |
| `test_adverse_shift` | `AdverseShiftResult` | `.statistic`, `.pvalue`, `.direction` |

## Primary API

- `test_shift(*, reference, candidate, statistic="roc_auc")`
- `test_adverse_shift(*, reference, candidate, direction=...)`

All arguments are keyword-only — you must name them when calling the functions.

Built-in statistics for `test_shift`:

- `roc_auc` (default, works for any numeric score)
- `balanced_accuracy` (binary scores only)
- `matthews_corrcoef` (binary scores only)

## Other modules

These modules are available when you need them, but you don't need them to get started:

| Module | What it does |
|--------|--------------|
| `samesame.advanced` | extra controls: sample weights, resampling depth, Bayesian test option |
| `samesame.logit_scores` | convert classifier outputs into confidence scores |
| `samesame.importance_weights` | build per-sample weights to account for group differences |
| `samesame.bayes_factors` | convert between p-values and Bayes factors |

Example advanced usage:

```python
from samesame import advanced

detail = advanced.test_shift(
	reference=os_train,
	candidate=os_test,
	n_resamples=4999,
)

harm_detail = advanced.test_adverse_shift(
	reference=os_train,
	candidate=os_test,
	direction="higher-is-worse",
	bayesian=True,
)
```

The advanced namespace returns detail-rich results, including resampling artifacts.

## Examples

Step-by-step examples are available in the [documentation](https://vathymut.github.io/samesame/):

**Tutorials**

- [Detect a distribution shift](https://vathymut.github.io/samesame/examples/distribution-shifts/)
- [Check whether a shift is harmful](https://vathymut.github.io/samesame/examples/noninferiority/)

**How-to guides**

- [Monitor a credit risk model](https://vathymut.github.io/samesame/examples/credit-example/)
- [Monitor model confidence](https://vathymut.github.io/samesame/examples/credit-ood-detection/)

## Dependencies

`samesame` has minimal dependencies. It is built on top of, and fully compatible with,
[scikit-learn][scikit-learn] and [numpy][numpy].

The public API is task-first, but the underlying methods still correspond to familiar ideas from
classifier two-sample testing and harmful-shift testing in the literature.

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
