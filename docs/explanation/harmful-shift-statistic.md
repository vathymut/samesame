# Why the harm statistic is not just AUC

`test_shift` and `test_harmful_shift` both compare a source group to a target group, and both run on a
permutation null. They answer different questions because they use different statistics.

- `test_shift` asks **did anything change?** It uses ROC AUC under a two-sided alternative.
- `test_harmful_shift` asks **did the target distribution shift toward worse outcomes?** It uses a directional,
  source-anchored statistic under a one-sided (`greater`) alternative.

This page explains what the harm statistic measures and why it is not redundant with AUC.

## The direction transform makes "worse" always mean "larger"

You declare the harmful direction with `worse`:

- `worse="higher"` — larger raw scores mean harm (e.g., predicted risk).
- `worse="lower"` — smaller raw scores mean harm (e.g., confidence, accuracy).

Internally, `worse="lower"` negates the scores. After that transform, **larger always means
worse**, regardless of which polarity you chose. Everything below assumes transformed scores.

## What the harm statistic integrates

Treat target as the positive class and let $S$ be the transformed score. As the threshold $t$ varies,
the ROC curve plots:

- $FPR(t) = P(S > t \mid \text{source})$ on the x-axis,
- $TPR(t) = P(S > t \mid \text{target})$ on the y-axis.

Notice that $1 - FPR(t) = P(S \le t \mid \text{source}) = \hat{F}_{\text{source}}(t)$, the source
empirical CDF. The harm statistic is

$$
T = \int TPR \cdot (1 - FPR)^{2} \, dFPR
  = \int TPR \cdot \hat{F}_{\text{source}}(t)^{2} \, dFPR.
$$

The weight $(1 - FPR)^{2}$ is largest at low $FPR$ — high thresholds that source rarely exceeds —
and falls to zero at $FPR = 1$. In other words, the statistic emphasises the part of the ROC where
target clears a high bar that source stays below.

## What that buys you

- **Directional.** If target has excess mass in the worse tail, $TPR$ stays high where $FPR$ is small,
  so $T$ is large. If target looks like source, $TPR \approx FPR$ throughout and $T$ is small.
- **Source-anchored.** Because the weight is built from $\hat{F}_{\text{source}}$, the comparison is
  calibrated to where the reference distribution lies, not to an arbitrary score range.
- **Less sensitive to symmetric change.** A shift that moves target *away* from source on the
  *better* side inflates two-sided AUC but does not inflate $T$ much, because the weight down at the
  high-$FPR$ end is tiny.

## How inference works

The null hypothesis is exchangeability: source and target are drawn from the same distribution, so
the group labels carry no information. `samesame` simulates that null by permuting the labels
`n_resamples` times and recomputing $T$ each time.

The p-value is one-sided (`alternative="greater"`): the fraction of null permutations whose $T$ is at
least as large as the observed one. A small p-value means the observed directional excess is unlikely
under no shift.

Permutation inference keeps the assumptions light — you do not need a parametric form for the score
distributions, only that the labels are exchangeable under the null.

## AUC vs. the harm statistic

| | `test_shift` (AUC) | `test_harmful_shift` (harm statistic) |
|---|---|---|
| Question | Did anything change? | Did the target distribution shift toward worse outcomes? |
| Statistic | $\int TPR \, dFPR$ | $\int TPR \cdot (1-FPR)^{2} \, dFPR$ |
| Weight on the ROC | uniform | favours low $FPR$ (the worse tail) |
| Alternative | two-sided | one-sided (`greater`) |
| Sensitive to | any separability | directional excess over source support |

Use `test_shift` when you only need to know whether the source and target distributions differ.
Use `test_harmful_shift` once you can declare what "worse" means for your signal. It will not flag a change
that shifts in a better or neutral direction.

For the worked examples, see
[Check whether target shifted toward worse outcomes](../examples/tutorials/check-shift-harm.md).
For the full API, see [Shift testing](../api/testing.md).
