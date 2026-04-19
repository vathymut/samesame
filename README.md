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

`samesame` helps you answer a question every data scientist faces after deploying a model:
**"Has my data changed in a way that could hurt my model?"**

It provides two complementary statistical tests:

- **CTST** — detects whether two datasets come from different distributions ("something changed")
- **DSOS** — detects whether that change is actually harmful ("things got worse")

This distinction matters. Not every distributional difference is a problem.
`samesame` helps you tell the two apart so you can avoid unnecessary alerts and focus on real issues.

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
from sklearn.metrics import roc_auc_score
from samesame.ctst import CTST
from samesame.nit import DSOS

rng = np.random.default_rng(123_456)
os_train = rng.normal(size=600)  # outlier scores from training
os_test  = rng.normal(size=600)  # outlier scores from deployment

# Question 1: Are the distributions different?
ctst = CTST.from_samples(os_train, os_test, metric=roc_auc_score)
print(f"CTST p-value: {ctst.pvalue:.4f}")
# CTST p-value: 0.0358  → distributions differ (small p-value)

# Question 2: Is the test set actually worse (more outliers)?
dsos = DSOS.from_samples(os_train, os_test)
print(f"DSOS p-value: {dsos.pvalue:.4f}")
# DSOS p-value: 0.9500  → no adverse shift detected (large p-value)
```

**What this means:** CTST flags a difference (the distributions are not identical), but DSOS says
the test set is *not* disproportionately worse. This is a common real-world situation — minor
statistical differences that do not signal a real problem. Without DSOS, you might raise a false alarm.

## Modules

| Module             | What it does                                              |
|--------------------|-----------------------------------------------------------|
| `samesame.ctst`    | Classifier two-sample tests — did the distribution change? |
| `samesame.nit`     | Noninferiority tests — is the change actually harmful?    |
| `samesame.bayes`   | Bayesian inference — convert p-values to Bayes factors    |
| `samesame.metrics` | Weighted ROC utilities (for example, WAUC)                |
| `samesame.ood`     | Out-of-distribution scoring — flag unusual inputs         |

## Test result attributes

Every test result object exposes these attributes (where applicable):

| Attribute       | Description                                      |
|-----------------|--------------------------------------------------|
| `.statistic`    | The observed test statistic                      |
| `.null`         | The null distribution (from permutations)        |
| `.pvalue`       | The p-value                                      |
| `.posterior`    | The Bayesian posterior distribution (DSOS only)  |
| `.bayes_factor` | The Bayes factor (DSOS only)                     |

## API quick reference

- **CTST**: Use `CTST.from_samples(a, b, metric=...)` for the common unweighted path, or
	`CTST(actual=..., predicted=..., metric=..., sample_weight=..., alternative=..., n_resamples=...)`
	when you need explicit control over weighting, hypothesis direction, or resampling depth.
- **DSOS / WeightedAUC**: `DSOS` is an alias of `WeightedAUC`.
	If you need `sample_weight`, construct `WeightedAUC(...)` directly.
- **OOD utilities**: `samesame.ood` includes both `logit_gap` (recommended default) and
	`max_logit` (simple baseline).

## Examples

Step-by-step worked examples are available in the [documentation](https://vathymut.github.io/samesame/):

- [Detecting distribution shifts](https://vathymut.github.io/samesame/examples/distribution-shifts/)
- [Noninferiority testing](https://vathymut.github.io/samesame/examples/noninferiority/)
- [Credit risk: shift and degradation](https://vathymut.github.io/samesame/examples/credit-example/)
- [Credit OOD detection](https://vathymut.github.io/samesame/examples/credit-ood-detection/)

## Dependencies

`samesame` has minimal dependencies. It is built on top of, and fully compatible with,
[scikit-learn][scikit-learn] and [numpy][numpy].

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
