# Score

A scalar signal — one number per observation — that `samesame` tests. Raw features are turned into a score first; the package never sees `X`.

Three built-for families:

| Family | Example | `worse` | Needs labels? |
|--------|---------|---------|---------------|
| **Predicted risk** | `P(default|x)` | `higher` | no |
| **Prediction error** | Brier ` (y-p)²`, log-loss | `higher` | yes |
| **Outlier score** | `LogitGap` confidence | `lower` (confidence) / `higher` (atypicality) | no |

Package term is *outlier score* (not “anomaly score” or “OOD score”). Confidence (`LogitGap`) is an outlier score where higher = more in-distribution / more certain.

Any scalar works — business KPI, latency, custom risk. See [Shift testing](../../api/testing.md#choose-a-function) for how the test uses it.

Domain probability `P(target|x)` can be a score for `test_shift`, but don't reuse it as the harm score when you also weight — it belongs in `domain_weights`.

See also: [`worse`](worse.md), [Harmful shift](harmful-shift.md), [Honest scores](honest-scores.md).
