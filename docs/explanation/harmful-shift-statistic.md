# Why the harm statistic is not just AUC

`test_shift` and `test_harmful_shift` share a permutation null but use different statistics for different questions.

- `test_shift` — **did anything change?** Two-sided ROC AUC.
- `test_harmful_shift` — **did target move toward worse outcomes?** One-sided, source-anchored.

## At a glance

AUC averages separability uniformly across the ROC curve. The harm statistic emphasizes the **harmful tail** — thresholds that source rarely exceeds but target clears. If target piles mass there, harm grows; if it shifts elsewhere, it doesn't.

> **TL;DR** — `AUC = ∫ TPR dFPR` (uniform). Harm = `∫ TPR·(1-FPR)² dFPR` (favours low `FPR`). Same null, different weighting.

## Direction: `worse` always means "larger" after the transform

Declare the harmful direction once (string or `ss.Worse`, interchangeable; enum gives autocomplete):

--8<-- "snippets/worse-table.txt"

Internally `polarity = scores if worse == "higher" else -scores` so larger always means worse. Everything below assumes transformed scores.

## What the harm statistic integrates

Treat target as the positive class and let `S` be the transformed score. For threshold `t`:

- `FPR(t) = P(S > t | source)` — x-axis
- `TPR(t) = P(S > t | target)` — y-axis

Since `1 - FPR(t) = P(S ≤ t | source) = F̂_source(t)` (source ECDF), the harm statistic is:

$$
T = \int TPR \cdot (1 - FPR)^{2} \, dFPR
  = \int TPR \cdot \hat{F}_{\text{source}}(t)^{2} \, dFPR.
$$

- `∫ TPR dFPR` = AUC (uniform).
- `∫ TPR·(1-FPR)² dFPR` = harm (weight `(1-FPR)²` favours low `FPR`).

`(1-FPR)²` peaks at low `FPR` — high thresholds source rarely exceeds — and decays to `0` at `FPR=1`. Target that clears a high bar source stays below gets a large `T`.

### Visual intuition

<div style="display:flex; gap:24px; flex-wrap:wrap; justify-content:center; margin:16px 0;">

<div style="flex:1; min-width:260px; max-width:380px; text-align:center;">

**Harmful shift** — excess mass at low `FPR`

<svg viewBox="0 0 220 180" width="100%" style="max-width:320px; border:1px solid #e5e7eb; border-radius:8px; background:white;">
  <!-- axes -->
  <line x1="30" y1="150" x2="190" y2="150" stroke="#111" stroke-width="1.5"/>
  <line x1="30" y1="150" x2="30" y2="20" stroke="#111" stroke-width="1.5"/>
  <text x="105" y="172" font-size="11" fill="#555" text-anchor="middle">FPR →</text>
  <text x="12" y="85" font-size="11" fill="#555" text-anchor="middle" transform="rotate(-90 12 85)">TPR ↑</text>
  <!-- heavily weighted region -->
  <rect x="30" y="20" width="48" height="130" fill="#3b82f6" opacity="0.12"/>
  <text x="54" y="35" font-size="8" fill="#1d4ed8" text-anchor="middle">(1−FPR)² large</text>
  <!-- diagonal -->
  <line x1="30" y1="150" x2="190" y2="20" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>
  <!-- harmful ROC: steep early rise -->
  <path d="M 30 150 C 50 70, 60 30, 190 20" fill="none" stroke="#1d4ed8" stroke-width="2.5"/>
  <circle cx="68" cy="42" r="4" fill="#1d4ed8"/>
  <!-- weight indicator -->
  <text x="68" y="58" font-size="7" fill="#1d4ed8" text-anchor="middle">heavily</text>
  <text x="68" y="66" font-size="7" fill="#1d4ed8" text-anchor="middle">weighted</text>
  <text x="30" y="165" font-size="9" fill="#6b7280">0</text>
  <text x="185" y="165" font-size="9" fill="#6b7280">1</text>
</svg>

<div style="font-size:12px; color:#1e40af; margin-top:6px;"><strong>Harm statistic: LARGE</strong> &middot; AUC: large</div>
</div>

<div style="flex:1; min-width:260px; max-width:380px; text-align:center;">

**Beneficial / symmetric shift** — mass at high `FPR`

<svg viewBox="0 0 220 180" width="100%" style="max-width:320px; border:1px solid #e5e7eb; border-radius:8px; background:white;">
  <line x1="30" y1="150" x2="190" y2="150" stroke="#111" stroke-width="1.5"/>
  <line x1="30" y1="150" x2="30" y2="20" stroke="#111" stroke-width="1.5"/>
  <text x="105" y="172" font-size="11" fill="#555" text-anchor="middle">FPR →</text>
  <text x="12" y="85" font-size="11" fill="#555" text-anchor="middle" transform="rotate(-90 12 85)">TPR ↑</text>
  <!-- lightly weighted region -->
  <rect x="30" y="20" width="48" height="130" fill="#9ca3af" opacity="0.07"/>
  <text x="54" y="35" font-size="8" fill="#6b7280" text-anchor="middle">(1−FPR)² small</text>
  <line x1="30" y1="150" x2="190" y2="20" stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>
  <!-- beneficial ROC: late rise -->
  <path d="M 30 150 C 70 140, 140 130, 190 20" fill="none" stroke="#6b7280" stroke-width="2.5"/>
  <text x="152" y="110" font-size="7" fill="#6b7280" text-anchor="middle">lightly</text>
  <text x="152" y="118" font-size="7" fill="#6b7280" text-anchor="middle">weighted</text>
  <text x="30" y="165" font-size="9" fill="#6b7280">0</text>
  <text x="185" y="165" font-size="9" fill="#6b7280">1</text>
</svg>

<div style="font-size:12px; color:#4b5563; margin-top:6px;"><strong>Harm statistic: small</strong> &middot; AUC: large (wrong side)</div>
</div>

</div>

A shift toward *better* outcomes inflates two-sided AUC but not `T` — the mass is at high `FPR` where the weight `≈ 0`. That's why `test_harmful_shift` is directional.

Computation is `O(n log n)` via `roc_curve` + trapezoidal rule.

## What that buys you

- **Directional.** Excess mass in the harmful tail → high `TPR` at low `FPR` → large `T`.
- **Source-anchored.** Weight is `F̂_source`, so the reference calibrates the comparison.
- **Robust to beneficial shift.** Movement on the better side inflates AUC but not `T`.

## How inference works

Scores and weights stay fixed — only labels are permuted.

- **Null:** exchangeability — source and target share the same distribution, so labels carry no information.
- **Procedure:** permute labels `n_resamples` times, recompute `T` each time.
- **p-value:** one-sided `greater` — fraction of null `T` at least as large as observed.

--8<-- "snippets/pvalue-guidance.txt"

No parametric form for scores is assumed. Weighted permutations permute the concatenated weight vector with labels, preserving group sizes.

Permutation p-values use +1 smoothing (Phipson & Smyth) and lie in `(0, 1]`; doubling for two-sided `test_shift` is capped at `1`.

--8<-- "snippets/n-resamples.txt"

??? tip "Plot the null"
    Compare `result.statistic` to `result.null_distribution`. If observed lies beyond `quantile(null, 0.95)`, `p < 0.05`. For AUC, `0.5` is chance; for the harm statistic, compare to the null median — no fixed `0.5` reference.

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

Use `test_shift` for "are they different?" Use `test_harmful_shift` once you can declare `worse` (`worse="higher"` / `ss.Worse.HIGHER` or `worse="lower"` / `ss.Worse.LOWER`). It won't flag neutral or beneficial shifts.

For a worked example, see [Test whether the shift is harmful](../examples/tutorials/check-shift-harm.md). For the API, see [Shift testing](../api/testing.md).

## References

- Kamulete, V. M., et al. (2022). Detecting harmful distribution shifts via scoring rules. *UAI*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
