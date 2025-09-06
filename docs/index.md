<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->
<!-- Add to badges above once on PyPI
    [![ReadTheDocs](https://readthedocs.org/projects/samesame/badge/?version=latest)](https://samesame.readthedocs.io/en/stable/)
    [![PyPI-Server](https://img.shields.io/pypi/v/samesame.svg)](https://pypi.org/project/samesame/)
    [![Monthly Downloads](https://pepy.tech/badge/samesame/month)](https://pepy.tech/project/samesame)
-->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: LGPLv3](https://img.shields.io/badge/License-LGPL--3.0-green.svg)](https://opensource.org/license/lgpl-3-0)
[![arXiv](https://img.shields.io/badge/arXiv-2107.02990-b31b1b.svg)](https://arxiv.org/abs/2107.02990) 

<!-- badges: end -->

> Same, same but different ...

`samesame` implements classifier two-sample tests (CTSTs) and as a bonus 
extension, a noninferiority test (NIT). 

These were either missing or implemented with some tradeoffs 
(looking at you, sample-splitting) in existing libraries. And so, 
`samesame` fills in the gaps :)

## Motivation

What is `samesame` good for? It is for data (model) validation, performance
monitoring, drift detection (dataset shift), statistical process control, and
on and on. 

Want more? 
[Here you go](https://vathymut.github.io/dsos/articles/motivation.html).
This motivating example comes from the related R package 
[`dsos`](https://github.com/vathymut/dsos).


## Installation

To install, run the following command:

```bash
python -m pip install samesame
```

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

To get started, please see the documentation and examples.

## Dependencies

`samesame` has few dependencies beyond the standard library. It will 
probably work with some older Python versions. It is, in short, a lightweight
dependency for most machine learning projects.`samesame` is built on top of,
and is compatible with, [scikit-learn][scikit-learn] and [numpy][numpy].

[numpy]: https://numpy.org/
[PyPI]: https://pypi.org/project/samesame
[scikit-learn]: https://scikit-learn.org/stable
