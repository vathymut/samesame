<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/pysmatch)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> Same, same but different ...

`samesame` implements classifier two-sample tests (CTSTs) and as a bonus extension, a noninferiority
test (NIT). These tests are either missing or implemented with significant tradeoffs (looking at you, sample-splitting) in existing libraries.

`samesame` is versatile, extensible, lightweight, powerful, and agnostic to your inference strategy
so long as it is valid (e.g. cross-fitting, sample splitting, etc.).

## Motivation

`samesame` is for those who need statistical tests for:

- **Data validation** - Verify that data distributions meet expectations
- **Model performance monitoring** - Detect performance degradation over time
- **Drift detection** - Identify dataset shifts between training and production
- **Statistical process control** - Monitor system behavior and quality
- [Covariate balance](https://kosukeimai.github.io/MatchIt/articles/assessing-balance.html) - Assess balance in observational studies

---

A [motivating example](https://vathymut.github.io/dsos/articles/motivation.html) is available from the related R package [`dsos`](https://github.com/vathymut/dsos), which provides some of the same functionality.

## Installation

To install, run the following command:

```bash
python -m pip install samesame
```

## Quick Start

This example demonstrates the key distinction between tests of equal distribution and noninferiority tests—a critical difference for avoiding false alarms in production systems.

Simulate outlier scores to test for no adverse shift:

```python
from samesame.ctst import CTST
from samesame.nit import DSOS
from sklearn.metrics import roc_auc_score
import numpy as np

n_size = 600
rng = np.random.default_rng(123_456)
os_train = rng.normal(size=n_size)
os_test = rng.normal(size=n_size)
null_ctst = CTST.from_samples(os_train, os_test, metric=roc_auc_score)
null_dsos = DSOS.from_samples(os_train, os_test)
```

**Test of equal distribution (CTST):** Rejects the null of equal distributions

```python
print(f"{null_ctst.pvalue=:.4f}")
# null_ctst.pvalue=0.0358
```

**Noninferiority test (DSOS):** Fails to reject the null of no adverse shift

```python
print(f"{null_dsos.pvalue=:.4f}")
# null_dsos.pvalue=0.9500
```

**Key insight:** While the test sample (`os_test`) has a statistically different distribution from the training sample (`os_train`), it does not contain disproportionally more outliers. This distinction is exactly what `samesame` highlights—many practitioners conflate "different distribution" with "problematic shift," but `samesame` helps you distinguish between the two.

## Usage

### Functionality

Below, you will find an overview of common modules in `samesame`.

| Function                                  | Module           |
|-------------------------------------------|------------------|
| Bayesian inference                        | `samesame.bayes` |
| Classifier two-sample tests (CTSTs)       | `samesame.ctst`  |
| Noninferiority tests (NITs)               | `samesame.nit`   |

### Attributes

When the method is a statistical test, `samesame` saves (stores) the results of
some potentially computationally intensive results in attributes. These
attributes, when available, can be accessed as follows.

| Attribute      | Description                                   |
|----------------|-----------------------------------------------|
| `.statistic`   | The test statistic for the hypothesis.        |
| `.null`        | The null distribution for the hypothesis.     |
| `.pvalue`      | The p-value for the hypothesis.               |
| `.posterior`   | The posterior distribution for the hypothesis.|
| `.bayes_factor`| The bayes factor for the hypothesis.          |

## Examples

To get started, please see the examples in the [docs](https://vathymut.github.io/samesame/).

## Dependencies

`samesame` has minimal dependencies beyond the Python standard library, making it a lightweight addition to most machine learning projects. It is built on top of, and fully compatible with, [scikit-learn][scikit-learn] and [numpy][numpy].

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
