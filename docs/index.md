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

Every monitoring question starts with the same check: *is this still like the reference?* The package name is the answer you hope to give. When the answer is no, the more important question is whether the difference moved toward worse outcomes.

Production monitoring often starts from a practical constraint: the raw feature space is too large to scan directly, and labels arrive too late to guide early action. A model score addresses this by reducing each observation to one interpretable number — predicted risk, prediction error, confidence, or an outlier score.

`samesame` compares that score between source and target.

--8<-- "snippets/source-target.txt"

It separates two questions that are easy to conflate:

- `ss.test_shift` — a broad, two-sided screen for any shift. Its statistic is the ROC AUC (`∫ TPR dFPR`); `0.5` means the score does not separate source from target.
- `ss.test_harmful_shift(..., worse="higher"|"lower")` — a focused, one-sided test for movement toward the tail you declare harmful. Its statistic is a weighted AUC (`∫ TPR·(1−FPR)² dFPR`) that emphasizes the harmful tail.

```python
--8<-- "snippets/quick-example.py:quick-example"
```

--8<-- "snippets/pvalue-caveat.txt"

## Workflow

1. **Choose a score** that captures the outcome you care about. If it comes from a fitted model, generate it out of sample.
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harmful_shift(..., worse=...)`.
4. **Address poor feature overlap** with `ss.domain_weights` only when it is a real concern. Weighting reframes the comparison around common support and is not a default correction.

## Where next

- **[Get started](examples/tutorials/get-started.md)** — 5 minutes, both tests.
- **[Is the new drug good enough?](examples/trials/check-drug-efficacy.md)** — the harm test through a trial of 70 scores, with no model.
- **[Monitor a credit model](examples/credit/monitor-credit.md)** — HELOC data: one model, three signals under the same shift.
- **[Weight for common support](examples/weighting/weight-for-common-support.md)** — when to reweight, and what it costs.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`. The package is designed for score-based monitoring and is not intended for randomized experiments, subgroup discovery, or sequential monitoring.
