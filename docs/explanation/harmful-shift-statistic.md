# Why the harm statistic is not just AUC

`test_shift` and `test_harmful_shift` both compare source vs target on a permutation null, but they answer different questions because they use different statistics.

- `test_shift` — **did anything change?** Two-sided ROC AUC.
- `test_harmful_shift` — **did target shift toward worse outcomes?** One-sided, source-anchored.

This page explains what the harm statistic measures and why it isn't redundant with AUC.

## At a glance

AUC averages separability uniformly across the ROC curve. The harm statistic weights the curve to emphasize the harmful tail — the high thresholds that source rarely exceeds but target clears. If target piles mass in that tail, the harm statistic grows; if target shifts elsewhere, it doesn't.

--8<-- "snippets/honest-scores.txt"

## The direction transform makes "worse" always mean "larger"

Declare the harmful direction with `worse`:

- `worse="higher"` — larger raw scores mean harm (e.g., risk). Used as is.
- `worse="lower"` — smaller raw scores mean harm (e.g., confidence). Scores are negated internally (`-scores`).

After the transform, **larger always means worse**, regardless of polarity. Everything below assumes transformed scores. `ss.Worse.HIGHER` / `ss.Worse.LOWER` are enum aliases.

```python
# inside samesame: src/samesame/shift.py:113
polarity = scores if worse == "higher" else -scores
```

## What the harm statistic integrates

Treat target as the positive class and let `S` be the transformed score. As threshold `t` varies, the ROC plots:

- `FPR(t) = P(S > t | source)` on the x-axis
- `TPR(t) = P(S > t | target)` on the y-axis

Note `1 - FPR(t) = P(S ≤ t | source) = F̂_source(t)`, the source empirical CDF. The harm statistic is:

$$
T = \int TPR \cdot (1 - FPR)^{2} \, dFPR
  = \int TPR \cdot \hat{F}_{\text{source}}(t)^{2} \, dFPR.
$$

- `∫ TPR dFPR` = AUC (uniform weight).
- `∫ TPR · (1-FPR)² dFPR` = harm statistic (weight `(1-FPR)²` favours low `FPR`).

The weight `(1-FPR)²` is largest at low `FPR` — high thresholds source rarely exceeds — and falls to zero at `FPR=1`. Target that clears a high bar source stays below gets a large `T`.

### Visual intuition

```
ROC for a harmful shift          ROC for a beneficial shift
(upper-left bulge)               (lower-right bulge)

TPR ↑  ╭──●                      TPR ↑  ╭
      ╱ │ heavily weighted       ╱ ─┤  lightly weighted
     ╱  │ (1-FPR)² large         ╱   │  (1-FPR)² small
    ╱   │                       ╱    │
   ╱────┘                      ╱─────╯
  └──────────→ FPR             └──────────→ FPR
   0    1                       0    1

Harm statistic: LARGE            Harm statistic: small
AUC: large                       AUC: large (but on the wrong side)
```

A shift toward *better* outcomes (lower transformed scores) inflates two-sided AUC but not `T`, because the action is at high `FPR` where the weight is tiny. That's why `test_harmful_shift` is directional.

Computation is `O(n log n)` via `roc_curve` + trapezoidal rule (`src/samesame/_statistics.py:11`).

## What that buys you

- **Directional.** Excess mass in the harmful tail → `TPR` stays high where `FPR` is small → large `T`. If `TPR ≈ FPR` throughout, `T` is small.
- **Source-anchored.** The weight is `F̂_source`, so the comparison is calibrated to where the reference distribution lies, not an arbitrary score range.
- **Less sensitive to symmetric/beneficial change.** A shift away from source on the *better* side inflates AUC but not `T`.

## How inference works

- **Null:** exchangeability — source and target come from the same distribution, so labels carry no information.
- **Procedure:** permute labels `n_resamples` times, recompute `T` each time. Scores (and, if given, importance weights) stay fixed; only labels move (`src/samesame/_permutation.py:132`).
- **p-value:** one-sided `greater` — fraction of null `T` at least as large as observed. Small p means the directional excess is unlikely under no shift.
- No parametric form for the score distributions is assumed — only exchangeability under the null. Weighted permutations permute the *concatenated* weight vector with labels, preserving the source/target weight lengths.

Permutation p-values use +1 smoothing (Phipson & Smyth) and are bounded in `(0, 1]`; doubling for two-sided `test_shift` is capped at `1`.

??? example "Weighted permutation in code"
    ```python
    # src/samesame/_permutation.py:119
    sample_weight = np.concatenate((weights.source, weights.target))
    observed = metric(labels, scores, sample_weight)
    for i in range(n_resamples):
        perm_labels = rng.permutation(labels)
        null[i] = metric(perm_labels, scores, sample_weight)
    ```

## AUC vs the harm statistic

|  | `test_shift` (AUC) | `test_harmful_shift` (harm) |
|---|---|---|
| Question | Did anything change? | Did target shift toward worse outcomes? |
| Statistic | `∫ TPR dFPR` | `∫ TPR·(1-FPR)² dFPR` |
| Weight on ROC | uniform | favours low `FPR` (harmful tail) |
| Alternative | two-sided | one-sided `greater` |
| Sensitive to | any separability | directional excess over source support |

Use `test_shift` when you only need "are they different?" Use `test_harmful_shift` once you can declare `worse`. It will not flag a shift in a better or neutral direction.

For a worked example, see [Check whether target shifted toward worse outcomes](../examples/tutorials/check-shift-harm.md). For the API, see [Shift testing](../api/testing.md).

## References

- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
