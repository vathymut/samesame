<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
<!-- badges: end -->

> *Same, same but different ...*

Bring your own score — the risk you care about, the errors your model makes, how much you trust it, or when something looks off. `samesame` tells you two things: did the source and target scores shift, and did it get worse?

Data and model monitoring rarely gives you labels, and univariate checks miss hidden multivariate (high-dimensional) shifts. A score reduces each observation to a single interpretable and insightful number.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`.

**Source** is the reference distribution (training data or a past deployment); **target** is the current deployment under evaluation.

It separates two questions that are easy to conflate:

- `ss.test_shift`: a broad, two-sided screen for any shift.
- `ss.test_harm(..., worse="higher")`: a focused, one-sided test for movement toward the tail you declare harmful (`worse="lower"` if that tail is the small one).

```python
import numpy as np

import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
harm = ss.test_harm(
    source=source_scores,
    target=target_scores,
    worse="higher",  # larger = more harm (e.g., risk)
    rng=rng,
)

print(f"Shift p-value: {shift.pvalue:.4f}")
# → Shift p-value: 0.0002
print(f"Harm  p-value: {harm.pvalue:.4f}")
# → Harm  p-value: 0.0001
```

The small p-values provide strong evidence that the target distribution shifted and moved in an adverse direction. We therefore reject both the null of no shift and the null of no harmful shift at any reasonable significance level.

## Workflow

1. **Choose a score** that represents the outcome you care about. Generate it out of sample if it comes from a fitted model.
2. **Ask whether anything changed** with `ss.test_shift`.
3. **Ask whether the change is harmful** with `ss.test_harm(..., worse=...)`. Specify in advance whether higher or lower scores indicate harm.
4. **Address poor feature overlap** with `ss.domain_weights` only when it is a real concern. Weighting focuses the comparison on a different population: the region of common support. Details: [Weight for common support](examples/credit/weight-for-common-support.md).

## Where next

- **[Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started/)**: run both tests in 5 minutes.
- **[Is the new drug good enough?](https://vathymut.github.io/samesame/examples/trials/check-drug-efficacy/)**: the harm test on 70 trial scores, no model.
- **[Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit/)**: one HELOC model, three signals.
- **[Weight for common support](examples/credit/weight-for-common-support.md)**: extend the HELOC monitoring example with common-support weighting.
- **[API reference](https://vathymut.github.io/samesame/api/testing/)**: full docs for the tests and `domain_weights`.
