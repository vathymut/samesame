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

Bring one score per observation (predicted risk, prediction error, or an outlier score such as a confidence gap). `samesame` compares it between source and target and answers two questions: did it change, and did it get worse?

Production monitoring is often constrained. The feature space is too wide to scan directly, and labels arrive too late. A score reduces each row to one number you can test.

--8<-- "snippets/source-target.txt"

It separates two questions that are easy to conflate:

- `ss.test_shift`: a broad, two-sided screen for any shift. Its statistic is the ROC AUC (`∫ TPR dFPR`), and `0.5` means the score does not separate source from target.
- `ss.test_harmful_shift(..., worse="higher"|"lower")`: a focused, one-sided test for movement toward the tail you declare harmful. Its statistic is a weighted AUC (`∫ TPR·(1−FPR)² dFPR`) that emphasizes the harmful tail.

```python
--8<-- "snippets/quick-example.py:quick-example"
```

A small p-value rejects label exchangeability, not impact, causality, effect size, or the chance the null is true. Evidence of a shift is not evidence of harm. Details: [Core concepts](explanation/core-concepts.md).

## Workflow

1. **Choose a score** that captures the outcome you care about. If a fitted model produces the scores, generate them out of sample ([Core concepts](explanation/core-concepts.md)).
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harmful_shift(..., worse=...)`.
4. **Address poor overlap** with `ss.domain_weights` only when overlap is poor. Weighting reframes around common support, not a default correction.

## Where next

- **[Get started](examples/tutorials/get-started.md)**: run both tests in 5 minutes.
- **[Is the new drug good enough?](examples/trials/check-drug-efficacy.md)**: the harm test on 70 trial scores, no model.
- **[Monitor a credit model](examples/credit/monitor-credit.md)**: one HELOC model, three signals.
- **[Weight for common support](how-to/weight-for-common-support.md)**: when and how to reweight.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`.

!!! note "Scope"
    Score-based monitoring only. Not for randomized experiments, subgroup discovery, or sequential monitoring.
