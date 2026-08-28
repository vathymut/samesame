# Harmful shift

A *directional* distributional change where target carries excess mass in the harmful tail you declared with `worse`.

- **Any shift** (`test_shift`, two-sided AUC): did distributions differ anywhere?
- **Harmful shift** (`test_harmful_shift`, one-sided): did target move toward worse outcomes?

```python
import samesame as ss
ss.test_harmful_shift(source=source_scores, target=target_scores, worse="higher", rng=12345)
```

If `target` piles mass where `source` rarely goes (low `FPR` in ROC terms), the harm statistic `∫ TPR·(1-FPR)² dFPR` grows. Mass on the beneficial side inflates AUC but not harm.

Not a synonym for “target got worse on average” — it is a tail property anchored to `source`’s support. See [Why the harm statistic is not just AUC](../harmful-shift-statistic.md) for the ROC weighting.

See also: [`worse`](worse.md), [Common support](common-support.md), [Permutation test](permutation-test.md).
