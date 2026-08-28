# Why the harm statistic is not just AUC

`test_shift` and `test_harmful_shift` share a permutation null but answer different questions because they use different statistics.

- `test_shift` — **did anything change?** Two-sided ROC AUC.
- `test_harmful_shift` — **did target move toward worse outcomes?** One-sided, source-anchored.

This page explains what the harm statistic measures and why it isn't redundant with AUC.

## At a glance

AUC averages separability uniformly across the ROC curve. The harm statistic weights the curve to emphasize the **harmful tail** — high thresholds that source rarely exceeds but target clears. If target piles mass there, the harm statistic grows; if target shifts elsewhere, it doesn't.

> **TL;DR** — `AUC = ∫ TPR dFPR` (uniform). Harm = `∫ TPR·(1-FPR)² dFPR` (favours low `FPR`, the harmful tail). Same permutation null, different weighting.

## Direction: `worse` always means "larger" after the transform

Declare the harmful direction once:

- `worse="higher"` (or `ss.Worse.HIGHER`) — larger raw scores mean harm (risk, error, atypicality). Used as is.
- `worse="lower"` (or `ss.Worse.LOWER`) — smaller raw scores mean harm (confidence, accuracy). Scores are negated internally (`-scores`).

After the transform, larger always means worse. Everything below assumes transformed scores.

```python
# inside samesame (src/samesame/shift.py)
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

`(1-FPR)²` is largest at low `FPR` — high thresholds source rarely exceeds — and decays to zero at `FPR=1`. Target that clears a high bar source stays below gets a large `T`.

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

Computation is `O(n log n)` via `roc_curve` + trapezoidal rule (`src/samesame/_statistics.py`).

## What that buys you

- **Directional.** Excess mass in the harmful tail → `TPR` stays high where `FPR` is small → large `T`. If `TPR ≈ FPR` throughout, `T` is small.
- **Source-anchored.** The weight is `F̂_source`, so the reference distribution calibrates the comparison, not an arbitrary score range.
- **Less sensitive to symmetric or beneficial change.** A shift away from source on the *better* side inflates AUC but not `T`.

## How inference works

- **Null:** exchangeability — source and target come from the same distribution, so labels carry no information.
- **Procedure:** permute labels `n_resamples` times, recompute `T` each time. Scores (and importance weights, if given) stay fixed; only labels move (`src/samesame/_permutation.py`).
- **p-value:** one-sided `greater` — fraction of null `T` at least as large as observed. Small p (typically ≤ 0.05) means the directional excess is unlikely under no shift.
- No parametric form for the scores is assumed — only exchangeability under the null. Weighted permutations permute the *concatenated* weight vector with labels, preserving the source/target lengths.

Permutation p-values use +1 smoothing (Phipson & Smyth) and lie in `(0, 1]`; doubling for two-sided `test_shift` is capped at `1`.

??? example "Weighted permutation in code"
    ```python
    # src/samesame/_permutation.py
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

Use `test_shift` when you only need "are they different?" Use `test_harmful_shift` once you can declare `worse` (`worse="higher"` / `ss.Worse.HIGHER` or `worse="lower"` / `ss.Worse.LOWER`). It will not flag a shift that is better or neutral.

For a worked example, see [Test whether the shift is harmful](../examples/tutorials/check-shift-harm.md). For the API, see [Shift testing](../api/testing.md).

## References

- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
