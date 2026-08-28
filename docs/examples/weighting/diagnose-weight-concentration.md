# How to: Diagnose weight concentration with effective sample size

Use this after building importance weights to check whether they are dispersed enough for a trustworthy test.

Weights correct for low overlap, but they can pile mass on a handful of observations. Then the weighted test compares a few points, not the whole group. Kish's effective sample size (ESS) measures that concentration *before* you interpret the result.

ESS ≤ `n`. Uniform weights give `ESS = n`; concentrated weights push ESS toward `1`.

!!! info "Prerequisites"
    - [Focus harmful-shift testing on common support](source-reweighting.md) — same HELOC setup, or run standalone with the synthetic example below.
    - [When importance weights help](../../explanation/importance-weights-rationale.md) — `shrinkage` intuition.

## What you need

- `source_prob` and `target_prob` — domain probabilities `P(target | x)` (see [Focus harmful-shift testing on common support](source-reweighting.md) for the full setup, or use the synthetic `beta` example below — no HELOC needed)
- a reason to suspect concentration: low overlap, or `shrinkage` near `0.0`

## Step 1 — Build the weights

Synthetic and self-contained so you can run it without HELOC.

```python
import numpy as np

import samesame as ss

rng = np.random.default_rng(7)

source_prob = rng.beta(a=2, b=5, size=400)
target_prob = rng.beta(a=5, b=2, size=400)

# A few source points look very target-like. At shrinkage=0 these dominate.
source_prob[:8] = rng.uniform(0.97, 0.999, size=8)

weights = ss.domain_weights(
    source=source_prob,
    target=target_prob,
    reweight="both",
    shrinkage=0.0,
)
```

--8<-- "snippets/clipping-note.txt"

## Step 2 — Read effective sample size

```python
ess = weights.effective_sample_size()

print(f"source ESS: {ess.source:.1f} / {len(source_prob)}")  # → 6.5 / 400
print(f"target ESS: {ess.target:.1f} / {len(target_prob)}")  # → 203.6 / 400
```

In this example source ESS collapses to ~`6` of `400` — a handful of source points carry almost all weight.

## Step 3 — Sweep `shrinkage` (λ)

ESS rises with `shrinkage`. Use the sweep to choose the smallest correction you can support.

```python
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    w = ss.domain_weights(
        source=source_prob,
        target=target_prob,
        reweight="both",
        shrinkage=lam,
    )
    e = w.effective_sample_size()
    print(
        f"shrinkage={lam:<4}  source ESS={e.source:7.2f}  "
        f"target ESS={e.target:7.2f}"
    )
```

```text
shrinkage=0.0   source ESS=   6.53  target ESS= 203.61
shrinkage=0.25  source ESS= 189.14  target ESS= 258.52
shrinkage=0.5   source ESS= 283.68  target ESS= 302.34
shrinkage=0.75  source ESS= 342.03  target ESS= 347.07
shrinkage=1.0   source ESS= 400.00  target ESS= 400.00
```

## How to read it

--8<-- "snippets/ess-rule.txt"

- Compare ESS to `n` for *each* group, not across groups.
- The collapse from `400` to `6.53` at `shrinkage=0.0` shows plain density-ratio weights are highly concentrated here — one source observation carries weight > `100`.
- `shrinkage=0.5` recovers most of the sample while still correcting for overlap — that's why it's the default.
- If ESS stays low even at `0.5`, groups barely overlap. Skip weighting or collect more comparable data rather than pushing `shrinkage` higher.

ESS is diagnostic, not a verdict. Use it with the weighted p-value: significant with healthy ESS is convincing; significant with `ESS ≈ 1` may be driven by one or two points. See [Glossary: ESS](../../explanation/glossary.md#effective-sample-size-ess).

??? example "Full script — copy and run (synthetic, no HELOC needed)"
    ```python
    --8<-- "examples/weighting/_code/diagnose_weight_concentration_example.py:full"
    ```

For formulas, see [When importance weights help](../../explanation/importance-weights-rationale.md). For the API, see [Importance weights](../../api/weighting.md).

??? tip "What next?"
    If ESS is healthy, interpret the weighted `test_harmful_shift` p-value. If not, try `shrinkage=0.5` → `0.75`, or `reweight="source"` only, and compare the three views in [Restrict to common support on both sides](double-weighting.md).
