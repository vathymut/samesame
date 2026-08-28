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

**Did the target shift? Did it get worse?**

Feature-level monitoring can produce alerts that are difficult to interpret,
and labels may arrive too late to support early action. `samesame` focuses
monitoring on a meaningful score per observation - such as predicted risk,
prediction error, confidence, or an outlier score - and compares it between
**source** (the reference) and **target** (the current deployment).

It separates two questions:

- **Any shift?** Use `ss.test_shift` to ask whether the score distribution changed between source and target. This is a two-sided AUC test.
- **Harmful shift?** Use `ss.test_harmful_shift(..., worse="higher"|"lower")` to ask whether the target moved in a specified harmful direction. This is a one-sided test based on the weighted AUC.

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
print(f"Harm  statistic: {harm.statistic:.3f}, p-value: {harm.pvalue:.4f}")
```

Interpret p-values alongside the statistic and score distributions: evidence of
a shift is not evidence of harm, and statistical significance is not business
impact.

> **Toy scores vs. real scores:** The example uses synthetic normal scores for
> brevity. For real features, build the score with a domain classifier and
> generate it out of sample - using `cross_val_predict`,
> `oob_decision_function_`, or held-out data - to avoid in-sample bias. See
> [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md).

## Workflow

1. Build **one score per observation**, generating it out of sample if it comes from a fitted model.
2. Test for **any shift** with `ss.test_shift`.
3. Test for **harmful shift** with `ss.test_harmful_shift(..., worse=...)`.
4. If source and target have poor overlap, use `ss.domain_weights` to reweight the comparison.

**Explore the documentation:**

- [Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started.md) - learn the workflow and run both tests in five minutes.
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit.md) - work through risk, confidence, and error monitoring with HELOC data.
- [Weight for common support](https://vathymut.github.io/samesame/examples/weighting/weight-for-common-support.md) - learn when and how to reweight comparisons.

[Read the full documentation](https://vathymut.github.io/samesame/).

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`.

For supported use cases and limitations, see the [full documentation](https://vathymut.github.io/samesame/).
