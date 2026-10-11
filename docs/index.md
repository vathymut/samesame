<!-- markdownlint-disable MD041 MD033 -->
# samesame

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)

> *Same, same but different...* — did deployment change, and did it get worse?

When labels are missing or delayed, reduce each observation to **one score you care about** — risk, error, confidence, outlier score — and compare **source** (reference: training / past deployment) vs **target** (current deployment).

Two questions, two tests:

- **Changed at all?** `ss.test_shift` — broad two-sided screen for any shift.
- **Got worse?** `ss.test_harm(..., worse="higher")` — focused one-sided test toward the tail you declare harmful *in advance*.

## Installation

```bash
python -m pip install samesame
```

Requires Python 3.12+, `numpy`, `scipy`, `scikit-learn`.

## Quickstart in 30s

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(12345)
source_scores = rng.normal(loc=0.0, scale=1.0, size=600)
target_scores = rng.normal(loc=0.6, scale=1.0, size=600)

shift = ss.test_shift(source=source_scores, target=target_scores, rng=rng)
harm = ss.test_harm(source=source_scores, target=target_scores, worse="higher", rng=rng)

print(f"Shift p-value: {shift.pvalue:.4f}")  # → 0.0002
print(f"Harm  p-value: {harm.pvalue:.4f}")   # → 0.0001
```

Small p-values = evidence against exchangeability — target shifted and got worse.

> Pick the score from meaning, not p-values. If it comes from a fitted model, generate it out of sample.

## Where next

- [Get started in 5 min](https://vathymut.github.io/samesame/examples/tutorials/get-started/)
- [Is the new drug good enough?](https://vathymut.github.io/samesame/examples/trials/check-drug-efficacy/) — harm test, no model, n=70
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit/) + [common-support weighting](https://vathymut.github.io/samesame/examples/credit/weight-for-common-support/)
- [API reference](https://vathymut.github.io/samesame/api/testing/) • [How the harm test works](https://vathymut.github.io/samesame/explanation/harmful-shift-statistic/)
