# How to: Diagnose weight concentration with effective sample size

Use this guide after you have built importance weights and want to know whether they are safe to use
in a shift test.

Importance weights correct for low overlap between source and target, but they can also put most of
their mass on a handful of observations. When that happens, the weighted test is effectively
comparing a few samples, not the whole group. Kish's effective sample size (ESS) tells you how bad
that is before you trust the result.

## What you need

- `source_prob` and `target_prob` from a domain classifier (see
  [Focus harmful-shift testing on shared support](source-reweighting.md) for the full setup)
- a reason to suspect concentration: low overlap, or `lambda_` set close to `0.0`

## Step 1 - Build the weights

This is the same call as in the other weighting guides. The example below is self-contained so you
can run it on its own.

```python
import numpy as np

from samesame.weights import from_domain_probabilities

rng = np.random.default_rng(7)

source_prob = rng.beta(a=2, b=5, size=400)
target_prob = rng.beta(a=5, b=2, size=400)

# A few source observations look very target-like. At lambda_=0 these drive
# the density ratio through the roof and dominate the weighted comparison.
source_prob[:8] = rng.uniform(0.97, 0.999, size=8)

weights = from_domain_probabilities(
    source_prob=source_prob,
    target_prob=target_prob,
    mode="both",
    lambda_=0.0,
)
```

## Step 2 - Read off the effective sample size

```python
ess = weights.effective_sample_size()

print(f"source ESS: {ess.source:.1f} / {len(source_prob)}")
print(f"target ESS: {ess.target:.1f} / {len(target_prob)}")
```

ESS is bounded above by the actual sample size. Uniform weights give `ESS = n`; a few observations
hogging all the weight push ESS toward 1. In this example, source ESS collapses to roughly `6` out
of `400`, which means a handful of source observations are doing almost all the work.

## Step 3 - Sweep `lambda_` and pick the smallest correction you trust

ESS rises with `lambda_`, so use it to find the trade-off you are willing to accept.

```python
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w = from_domain_probabilities(
        source_prob=source_prob,
        target_prob=target_prob,
        mode="both",
        lambda_=lam,
    )
    e = w.effective_sample_size()
    print(
        f"lambda_={lam:<4}  source ESS={e.source:7.2f}  "
        f"target ESS={e.target:7.2f}"
    )
```

```
lambda_=0.0   source ESS=   6.53  target ESS= 203.61
lambda_=0.25  source ESS= 189.14  target ESS= 258.52
lambda_=0.5   source ESS= 283.68  target ESS= 302.34
lambda_=0.75  source ESS= 342.03  target ESS= 347.07
lambda_=1.0   source ESS= 400.00  target ESS= 400.00
```

## How to read the result

- Compare ESS to `n` for each group, not to each other. A useful rule of thumb: worry when ESS is a
  small fraction of `n` (say, below a quarter).
- The collapse from `400` to `6.53` at `lambda_=0.0` is the signal that plain density-ratio weights
  are unsafe here — one source observation carries a weight above `100`.
- `lambda_=0.5` recovers most of the sample while still correcting for overlap. That is why it is the
  default.
- If ESS stays low even at `lambda_=0.5`, the groups barely overlap. Consider skipping weights
  altogether or collecting more comparable data rather than pushing `lambda_` higher.

ESS is a diagnostic, not a verdict. Use it alongside the weighted p-value: a significant weighted
result with a healthy ESS is meaningful; a significant weighted result with `ESS ≈ 1` is an artifact
of one or two observations.

For the intuition behind the weighting formulas, see
[When importance weights help](../../explanation/importance-weights-rationale.md).
For the API, see [Importance weights](../../api/weighting.md).
