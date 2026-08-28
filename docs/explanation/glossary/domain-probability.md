# Domain probability

`P(target | x)` — output of a domain classifier trained to distinguish source from target.

Passed as **two separate 1-D arrays** to `domain_weights`:

```python
weights = ss.domain_weights(
    source=source_prob,  # P(target|x) for source rows, length n_source
    target=target_prob,  # P(target|x) for target rows, length n_target
    reweight="both",
)
```

- Prior ratio `n_source / n_target` is inferred from lengths, not tuned.
- Probabilities at `0` / `1` are clipped to `[1e-6, 1-1e-6]` before ratios (see [Importance weights help](../importance-weights-rationale.md)).
- Use cross-validated or OOB predictions — don't train and score on the same rows.

**Role:** building weights, not a harm signal. It can be the score for `test_shift` (detect any change), but don't reuse it as the harm score when you also weight — weighting already uses it.

See also: [Importance weights](importance-weights.md), [Common support](common-support.md), [Reweight](reweight.md).
