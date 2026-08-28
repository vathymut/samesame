# `worse`

Polarity that declares which direction is harmful. Required once per `test_harmful_shift`.

| Signal | Worse when | Use |
|--------|------------|-----|
| Predicted risk, error, atypicality | larger | `worse="higher"` (or `ss.Worse.HIGHER`) |
| Confidence, accuracy, quality | smaller | `worse="lower"` (or `ss.Worse.LOWER`) |

String or `ss.Worse` — interchangeable; enum gives autocomplete.

Internally `polarity = scores if worse == "higher" else -scores`, so larger always means worse. Everything in [Why the harm statistic is not just AUC](../harmful-shift-statistic.md) assumes transformed scores.

```python
import samesame as ss
ss.test_harmful_shift(source=src, target=tgt, worse="higher", rng=12345)
ss.test_harmful_shift(source=src, target=tgt, worse=ss.Worse.LOWER, rng=12345)
```

Choosing the wrong `worse` flips the harmful tail and can hide a real harmful shift. When direction is unclear, start with `test_shift` (two-sided).

See also: [Harmful shift](harmful-shift.md), [Score](score.md), [Shift testing](../../api/testing.md).
