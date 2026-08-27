# Tutorial: Check whether target shifted toward worse outcomes

Use this tutorial when you already have a signal with a clear direction and want to know whether the
target distribution shifted toward worse outcomes.

By the end, you will know how to:

- distinguish "different" from "worse"
- choose the correct `worse` direction
- run `ss.detect_harm(...)` and interpret the p-value

`ss.detect_shift(...)` asks whether source and target differ at all.
`ss.detect_harm(...)` asks a narrower question: did the target distribution shift toward the harmful
end of the signal?

## What you need

- a source group and a target group
- a numeric signal for each group
- a clear decision about whether larger values are better or worse

## Step 1 - Make a simple example

Imagine these values are model-quality or confidence scores, where **higher is better**.

```python
import numpy as np

rng = np.random.default_rng(123_456)

source_quality = rng.normal(loc=0.80, scale=0.07, size=400)
target_quality = rng.normal(loc=0.72, scale=0.07, size=400)
```

This example gives the target distribution slightly lower scores, so it illustrates a harmful shift.

## Step 2 - Compare any change with harmful change

```python
import samesame as ss

shift = ss.detect_shift(source_quality, target_quality)

harm = ss.detect_harm(
    source_quality,
    target_quality,
    worse="lower",
)

print(f"Shift p-value: {shift.pvalue:.4f}")
print(f"Harm  p-value: {harm.pvalue:.4f}")
```

Because higher values are better here, we use `worse="lower"`. That tells `samesame` to treat a
shift toward lower target scores as harmful.

## How to read the result

- A small p-value from `detect_shift(...)` means the groups differ.
- A small p-value from `detect_harm(...)` means the target distribution also shifted toward worse
  outcomes.
- If your signal already uses larger values for worse outcomes, use `worse="higher"` instead.

Typical examples of `higher-is-worse` signals are predicted default risk, error, or anomaly level.
Typical examples of `higher-is-better` signals are confidence, accuracy, or quality.

For the full API of both tests, see [Shift testing](../../api/testing.md).
When source and target do not cover the same feature space, continue to
[Focus on shared support with importance weights](adjust-for-covariate-shift.md).
