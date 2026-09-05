<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
<!-- badges: end -->

> *Same, same but different ...*

Every monitoring question starts as one word: *same?* Is the new sample the
same as the reference? The name is the answer you hope to give. The work —
and the reason this package exists — is the second half of the phrase: when
the target is different, is it different in the way that matters, **worse**?

Production monitoring usually starts with a practical problem: the raw feature
space is too large to interpret directly, and labels often arrive late. A model
score reduces each observation — predicted risk, prediction error, confidence,
or an outlier score — to a single interpretable number, one per row.

`samesame` compares that score between **source** (the reference, such as
training data or a past deployment) and **target** (the current deployment).
It separates two questions that should not be conflated:

- `ss.test_shift` — broad, two-sided screen for any shift (ROC AUC
  `∫ TPR dFPR`, `0.5` is chance).
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — focused, one-sided
  test for movement toward the tail you declare harmful (weighted AUC
  `∫ TPR·(1−FPR)² dFPR`, which emphasizes the harmful tail).

```python
--8<-- "snippets/quick-example.py:quick-example"
```

Read `.pvalue` as evidence against label exchangeability — not business
impact, causality, or the probability the null is true. Evidence of a shift
is not evidence of harm.

## Workflow

1. **Choose a score** that represents the outcome you care about. Generate it
   out of sample if it comes from a fitted model.
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harmful_shift(...,
   worse=...)`.
4. **Address poor feature overlap** with `ss.domain_weights` only when it is a
   real concern — weighting changes the population the comparison describes
   and is not a default correction.

## Where next

- **[Get started](examples/tutorials/get-started.md)** — 5 min, both tests.
- **[Is the new drug good enough?](examples/trials/check-drug-efficacy.md)** — the harm test, told as a trial with 70 scores and no model.
- **[Monitor a credit model](examples/credit/monitor-credit.md)** — HELOC: one model, three signals, one storm.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when to reweight, and what it costs.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`. Not for randomized experiments, subgroup discovery, or sequential monitoring.
