<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> *Same, same but different ...*

**Did the target shift? Did it get worse?**

Feature-level monitoring can produce alerts that are hard to interpret, and labels often arrive too late for early action. `samesame` takes a simpler route: reduce each observation to one interpretable score — predicted risk, prediction error, confidence, or an outlier score — and compare that score between **source** (the reference) and **target** (the current deployment). Each row contributes one number to monitor.

It separates two questions that are easy to conflate:

- **Any shift?** Use `ss.test_shift` to ask whether the score distribution differs between source and target. This is a broad, two-sided test based on the ROC AUC.
- **Harmful shift?** Use `ss.test_harmful_shift(..., worse="higher"|"lower")` to ask whether the target moved toward the tail you declare harmful. This is a focused, one-sided test based on a weighted AUC.

## Quick example

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
harm = ss.test_harmful_shift(
    source=source_scores,
    target=target_scores,
    worse="higher",  # larger = more harm (e.g., risk)
    rng=rng,
)

print(f"Shift statistic: {shift.statistic:.3f}, p-value: {shift.pvalue:.4f}")
# → Shift statistic: 0.697, p-value: 0.0002
print(f"Harm  statistic: {harm.statistic:.3f}, p-value: {harm.pvalue:.4f}")
# → Harm  statistic: 0.155, p-value: 0.0001
```

Interpret p-values alongside the statistic and the score distributions. A small p-value is evidence against label exchangeability (the assumption that source and target labels can be swapped). It is not evidence of business impact, causality, effect size, or the probability that the null is true. Evidence of a shift is not evidence of harm.

> **Toy scores versus real scores:** Synthetic normals keep the example brief. When a fitted model produces the scores, generate them out of sample with `cross_val_predict`, `oob_decision_function_`, or a held-out set. In-sample predictions use information the model has already seen; they can make source and target look more separable than they are and invalidate the test. See [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started/).

## Workflow

1. **Choose a score** that captures the outcome you care about. If it comes from a fitted model, generate it out of sample.
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harmful_shift(..., worse=...)`.
4. **Address poor feature overlap** with `ss.domain_weights` only when it is a real concern. Weighting reframes the comparison around common support and is not a default correction.

**Explore the documentation:**

- [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started/) — learn the workflow and run both tests in five minutes.
- [Is the new drug good enough?](https://vathymut.github.io/samesame/examples/trials/check-drug-efficacy/) — the harm test explained through a classic noninferiority trial, with no model required.
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit/) — work through risk, confidence, and error monitoring with HELOC data.
- [Weight for common support](https://vathymut.github.io/samesame/examples/weighting/weight-for-common-support/) — learn when and how to reweight comparisons.

[Read the full documentation](https://vathymut.github.io/samesame/).

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`. The package is designed for score-based monitoring and is not intended for randomized experiments, subgroup discovery, or sequential monitoring.

For supported use cases and limitations, see the [full documentation](https://vathymut.github.io/samesame/).
