<!-- markdownlint-disable MD041 MD033 -->
# samesame

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)

> *Same, same but different...* — did it change, and did it get worse?

Choose **one score you care about** for each observation: risk, error,
confidence, an outlier score, or the outcome itself. Compare **source**, the
reference sample (often training data or a past deployment), with **target**,
the sample under review (often a current deployment). Use a model score while
labels are delayed. When the outcome is available, use it as the score.

Two questions, two tests:

- **Do scores separate the groups?** `ss.test_shift` — a two-sided ROC AUC test
  of whether target scores tend to be higher or lower than source scores. It can
  miss changes in distribution shape that do not affect this ranking.
- **Got worse?** `ss.test_harm(..., worse="higher")` — focused one-sided test
  toward the tail you declare harmful *in advance*.

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

Both p-values are small. In this draw the target moved up, which is the direction marked harmful.

> Pick the score from meaning, not p-values. If it comes from a fitted model, generate it out of sample.

## Where next

- [Get started in 5 min](https://vathymut.github.io/samesame/examples/tutorials/get-started/)
- [Is the new drug good enough?](https://vathymut.github.io/samesame/examples/trials/check-drug-efficacy/) — harm test, no model, n=70
- [Monitor a credit model](https://vathymut.github.io/samesame/examples/credit/monitor-credit/) + [common-support weighting](https://vathymut.github.io/samesame/examples/credit/weight-for-common-support/)
- [Benign vs harmful shift](https://vathymut.github.io/samesame/explanation/benign-vs-harmful-shift/) · [How the harm test works](https://vathymut.github.io/samesame/explanation/harmful-shift-statistic/) · [API reference](https://vathymut.github.io/samesame/api/testing/)
