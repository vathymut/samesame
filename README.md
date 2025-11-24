<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/samesame)](https://pypi.org/project/samesame/)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![License: LGPLv3](https://img.shields.io/badge/License-LGPL--3.0-green.svg)](https://opensource.org/license/lgpl-3-0)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990) 
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> Same, same but different ...

`samesame` implements classifier two-sample tests (CTSTs) and as a bonus 
extension, a noninferiority test (NIT). 

These were either missing or implemented with some tradeoffs 
(looking at you, sample-splitting) in existing libraries. And so, 
`samesame` fills in the gaps :)

## Motivation

What is `samesame` good for? It is for data (model) validation, performance
monitoring, drift detection (dataset shift), statistical process control, 
[covariate balance](https://kosukeimai.github.io/MatchIt/articles/assessing-balance.html) 
and so on and so forth. 

As an example, this
[motivating example](https://vathymut.github.io/dsos/articles/motivation.html) 
comes from the related R package [`dsos`](https://github.com/vathymut/dsos).

## Installation

To install, run the following command:

```bash
python -m pip install samesame
```

## Quick Start

Simulate outlier scores to test for no adverse shift when the null (no
shift) holds. 

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

In this example, we reject the null of equal distribution (i.e. `CTST`)

```python
print(f"{null_ctst.pvalue=:.4f}")
# null_ctst.pvalue=0.0358
```

However, we fail to reject the null of no adverse shift (i.e. `DSOS`), meaning 
that the test sample (`os_test`) does not seem to contain disproportionally
more outliers than the training sample (`os_train`).

```python
print(f"{null_dsos.pvalue=:.4f}")
# null_dsos.pvalue=0.9500
```

This is the type of false alarms that `samesame` can highlight by comparing
tests of equal distribution to noninferiority tests.

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

`samesame` has few dependencies beyond the standard library. It will 
probably work with some older Python versions. It is, in short, a lightweight
dependency for most machine learning projects.`samesame` is built on top of,
and is compatible with, [scikit-learn][scikit-learn] and [numpy][numpy].

[numpy]: https://numpy.org/
[PyPI]: https://pypi.org/project/samesame
[scikit-learn]: https://scikit-learn.org/stable
