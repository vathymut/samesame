# `worse`

Polarity that declares which direction is harmful. Required once per `test_harmful_shift`.

| Signal | What it measures | `worse` |
|--------|------------------|---------|
| Predicted risk | business impact | `worse="higher"` |
| Prediction error (Brier, log-loss) | accuracy once labels arrive | `worse="higher"` |
| Outlier score — confidence (`LogitGap`) | certainty / typicality | `worse="lower"` |
| Outlier score — atypicality | distance from source | `worse="higher"` |

Strings and `ss.Worse` enum are interchangeable (`"higher"` ↔ `ss.Worse.HIGHER`); enum gives autocomplete.

Internally `polarity = scores if worse == "higher" else -scores`, so larger always means worse. Everything in [Why the harm statistic is not just AUC](../harmful-shift-statistic.md) assumes transformed scores.

```python
import samesame as ss
ss.test_harmful_shift(source=src, target=tgt, worse="higher", rng=12345)
ss.test_harmful_shift(source=src, target=tgt, worse=ss.Worse.LOWER, rng=12345)
```

Choosing the wrong `worse` flips the harmful tail and can hide a real harmful shift. When direction is unclear, start with `test_shift` (two-sided).

See also: [Harmful shift](harmful-shift.md), [Score](score.md), [Shift testing](../../api/testing.md).
