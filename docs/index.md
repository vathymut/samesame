<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
<!-- badges: end -->

**Did the target shift? Did it get worse?**

Production monitoring usually starts with a practical problem: the raw feature
space is too large to interpret directly, and labels often arrive late. A model
score gives each observation one meaningful number to monitor: predicted risk,
prediction error, confidence, or an outlier score.

`samesame` compares that score between **source** (the reference, such as
training data or a past deployment) and **target** (the current deployment).
It separates two questions that should not be conflated:

- `ss.test_shift` — can the score distinguish source from target at all?
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — is target unusually
  concentrated in the harmful direction you specify?

The first test is two-sided and reports ROC AUC. The second is one-sided and
weights the part of the score range where target exceeds thresholds that few
source observations exceed. This makes it sensitive to a harmful tail rather
than to every kind of distribution shift.

```python
--8<-- "snippets/quick-example.py:quick-example"
```

Read `.pvalue` as evidence against the relevant null, then inspect the
statistic and the score distributions for practical importance. A small
`test_shift` p-value means the groups differ, not that the difference is
harmful. A small harmful-shift p-value means the specified harmful direction
is supported; it does not measure business impact or prove causality.

## Workflow

1. **Choose a score** that represents the outcome you care about. Generate it
   out of sample if it comes from a fitted model.
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harmful_shift(...,
   worse=...)`.
4. **Address poor feature overlap** with `ss.domain_weights` only when it is a
   real concern; weighting changes the population the comparison describes.

## Where next

- **[Get started](examples/tutorials/get-started.md)** — 5 min, both tests.
- **[Monitor a credit model](examples/credit/monitor-credit.md)** — HELOC: risk, confidence, errors.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when to reweight.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`. Not for randomized experiments, subgroup discovery, or sequential alarming.
