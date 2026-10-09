<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
<!-- badges: end -->

> *Same, same but different ...*

For ML / risk teams monitoring systems/models when labels are missing or delayed. Bring your own score — the risk you care about, model errors, trust, or when something looks off. **Source** is the reference (training data or past deployment); **target** is the current deployment.

Works with or without labels. One score per observation catches multivariate shifts univariate checks miss. Two questions: did scores shift, and did it get worse?

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, and `scikit-learn`.

## Quickstart in 30s

Changed at all? `ss.test_shift`: broad, two-sided screen for any shift.
Got worse? `ss.test_harm(..., worse="higher")`: focused, one-sided test toward the tail you declare harmful. Declare `worse` in advance.

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

Small p-values = strong evidence the target shifted and got worse.

> Choose a score that captures the outcome you care about. If it comes from a fitted model, generate it out of sample.

## Where next

- **[Get started](https://vathymut.github.io/samesame/examples/tutorials/get-started/)**: run both tests in 5 minutes.
- **[Is the new drug good enough?](https://vathymut.github.io/samesame/examples/trials/check-drug-efficacy/)**: harm test on 70 trial scores, no model.
- **[Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit/)**: one HELOC model, three signals — see also [common-support weighting](https://vathymut.github.io/samesame/examples/credit/weight-for-common-support/).
- **[API reference](https://vathymut.github.io/samesame/api/testing/)**: full docs for the tests and `domain_weights`.

> Scores are univariate: declare `worse="higher" | "lower"` from meaning, not p-values. See [How the harm test works](https://vathymut.github.io/samesame/explanation/harmful-shift-statistic/).
