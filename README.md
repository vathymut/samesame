<!-- markdownlint-disable MD041 -->
<!-- markdownlint-disable MD033 -->

# samesame

<!-- badges: start -->
[![Development Status](https://img.shields.io/badge/status-early%20development-yellow)](https://github.com/vathymut/samesame)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://pypi.org/project/samesame/)
[![Downloads](https://static.pepy.tech/badge/samesame)](https://pepy.tech/project/samesame)
[![Static Badge](https://img.shields.io/badge/docs-link-blue)](https://vathymut.github.io/samesame/)
[![UAI 2022](https://img.shields.io/badge/paper-UAI%202022-yellow)](https://arxiv.org/abs/2107.02990)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<!-- badges: end -->

> Same, same but different ...

`samesame` is a Python package for comparing a reference population with a new population.
It helps answer two practical questions:

- Did anything change?
- Did the change move in a worse direction?

In the package, the reference group is called **source** and the new group is called **target**.
Typical examples are training vs production data, last month's batch vs this month's batch, or one
customer segment vs another.

You bring a numeric signal for each observation, and `samesame` handles the statistical testing.
That signal can be whatever best matches your use case: predicted risk, model confidence,
prediction error once labels arrive, or a classifier score used to compare two datasets.

The API stays small on purpose:

- Use `ss.shift.detect_shift(...)` to test whether source and target differ.
- Use `ss.shift.detect_harm(...)` to test whether target moved in a worse direction.
- Use `ss.weights.from_domain_probabilities(...)` when you want weighting that focuses the test on
  the region where source and target overlap.

## Who is this for?

`samesame` is a good fit when you already know what signal you want to monitor for each
observation, for example:

- **Model monitoring** - Does production still look like training, and are predictions getting worse?
- **Data validation** - Does this new batch still look like the data I expect?
- **Population comparison** - Do two customer groups, regions, or experiments behave differently?
- **Covariate-shift adjustment** - Can I focus on the region where both groups are genuinely comparable?

## Installation

```bash
python -m pip install samesame
```

## Quick Start

Suppose you already have a numeric signal for a source group and a target group.
Larger values can mean more unusual cases, more risk, less confidence, or any other notion that is
relevant to the question you are asking.

```python
import numpy as np
import samesame as ss

rng = np.random.default_rng(123_456)
source_scores = rng.normal(size=600)
target_scores = rng.normal(size=600)

shift = ss.shift.detect_shift(source_scores, target_scores)
print(f"Did anything change?  p-value = {shift.pvalue:.4f}")

harm = ss.shift.detect_harm(
    source_scores,
    target_scores,
    direction="higher-is-worse",
)
print(f"Did things get worse? p-value = {harm.pvalue:.4f}")
```

How to read this:

- A small p-value from `ss.shift.detect_shift(...)` means the target group looks different from the source group.
- A small p-value from `ss.shift.detect_harm(...)` means it also moved in the declared worse direction.
- If the first is small and the second is large, something changed, but not in a clearly harmful way.

## Choosing the signal

`samesame` does not decide for you what should count as "worse". That depends on your workflow.
Common choices are:

- **Predicted risk** when the model output already has business meaning.
- **Prediction error** when labels are available and you want a direct accuracy signal.
- **Confidence score** when you want to track how certain the model looks.
- **Domain-classifier probability** when your goal is to detect distribution shift between datasets.

The tutorials and how-to guides show how to choose and build these signals in practice.

## Why users reach for it

The methods in `samesame` are statistically grounded, but day-to-day usage stays simple: you pick a
signal, compare source with target, and read the result in terms of change and harm.

Both tests are permutation-based, so they do not depend on fragile parametric assumptions. When you
know that source and target differ in feature coverage, you can add importance weights to focus the
test on the region where the two groups overlap.

## Where to go next

Step-by-step examples are available in the [documentation](https://vathymut.github.io/samesame/):

**Tutorials**

- [Detect whether two datasets differ](https://vathymut.github.io/samesame/examples/tutorials/detect-distribution-shift/)
- [Check whether a change points in a worse direction](https://vathymut.github.io/samesame/examples/tutorials/check-shift-harm/)
- [Focus on shared support with importance weights](https://vathymut.github.io/samesame/examples/tutorials/adjust-for-covariate-shift/)

**How-to guides**

- [Monitor predicted credit risk](https://vathymut.github.io/samesame/examples/credit/monitor-credit-risk/)
- [Monitor model confidence](https://vathymut.github.io/samesame/examples/credit/monitor-model-confidence/)
- [Monitor prediction errors once labels arrive](https://vathymut.github.io/samesame/examples/credit/monitor-prediction-errors/)
- [Focus harmful-shift testing on shared support](https://vathymut.github.io/samesame/examples/weighting/source-reweighting/)
- [Restrict testing to common support on both sides](https://vathymut.github.io/samesame/examples/weighting/double-weighting/)

## Dependencies

`samesame` has minimal dependencies and fits naturally into a NumPy and scikit-learn workflow.

[numpy]: https://numpy.org/
[scikit-learn]: https://scikit-learn.org/stable
