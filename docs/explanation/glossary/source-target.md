# Source and target

Two groups of scores compared by every test in `samesame`.

- **Source** — reference distribution. Usually training data or a past batch. `source=` holds its scores.
- **Target** — evaluation distribution. Usually production or a new batch. `target=` holds its scores.

Both are one numeric score per observation (not raw feature tables). Raw features are reduced to a scalar — risk, error, or outlier score — before calling `test_shift` / `test_harmful_shift`.

```python
import samesame as ss
ss.test_shift(source=source_scores, target=target_scores, rng=12345)
```

Use `source=` / `target=` arguments everywhere; `group` / `membership_prob` are historic names.

**Novice:** think *training vs production*. **Expert:** exchangeability under the null assumes source and target are labelled draws from the same mixture; permutation tests the label.

See also: [Score](score.md), [Common support](common-support.md), [Glossary overview](../glossary.md).
