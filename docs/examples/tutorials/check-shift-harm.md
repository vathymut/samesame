# Tutorial: Check whether a change points in a worse direction

Use this tutorial when you already have a signal with a clear direction and want to know whether
the target group moved the wrong way.

By the end, you will know how to:

- distinguish "different" from "worse"
- choose the correct `direction`
- run `ss.shift.detect_harm(...)` and interpret the p-value

`ss.shift.detect_shift(...)` asks whether source and target differ at all.
`ss.shift.detect_harm(...)` asks a narrower question: did the target group pick up more of the bad
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

The target group is slightly worse on purpose, so this is a good example for learning the test.

## Step 2 - Compare any change with harmful change

```python
import samesame as ss

shift = ss.shift.detect_shift(source_quality, target_quality)

harm = ss.shift.detect_harm(
    source_quality,
    target_quality,
    direction="higher-is-better",
)

print(f"Shift p-value: {shift.pvalue:.4f}")
print(f"Harm  p-value: {harm.pvalue:.4f}")
```

Because higher values are better here, we use `direction="higher-is-better"`. That tells
`samesame` to treat a downward move in the target group as harmful.

## How to read the result

- A small p-value from `detect_shift(...)` means the groups differ.
- A small p-value from `detect_harm(...)` means the target group also moved in the worse direction.
- If your signal already uses larger values for worse outcomes, use `direction="higher-is-worse"` instead.

Typical examples of `higher-is-worse` signals are predicted default risk, error, or anomaly level.
Typical examples of `higher-is-better` signals are confidence, accuracy, or quality.

For posterior draws and a Bayes factor alongside the p-value, see
[Shift testing](../../api/testing.md).
