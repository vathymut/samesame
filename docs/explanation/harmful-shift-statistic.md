# How the harm test works

`test_shift` and `test_harmful_shift` share a permutation null but answer different questions.

- `test_shift` — did anything change? Two-sided AUC.
- `test_harmful_shift` — did target move toward worse outcomes? One-sided, source-anchored.

## At a glance

AUC averages separability uniformly. Harm emphasizes the **harmful tail** — thresholds source rarely exceeds but target clears.

> `AUC = ∫ TPR dFPR` (uniform) · Harm = `∫ TPR·(1-FPR)² dFPR` (favours low `FPR`).

If target piles mass in the harmful tail, harm grows; if it shifts elsewhere, it doesn't.

## Direction

Declare once — string or `ss.Worse` (interchangeable):

--8<-- "snippets/worse-table.txt"

Internally `scores if worse=="higher" else -scores` so larger always means worse.

??? details "What it integrates (experts) — the formula"

    Let `S` be the transformed score, target = positive class, threshold `t`:

    - `FPR(t)=P(S>t|source)` (x-axis)
    - `TPR(t)=P(S>t|target)` (y-axis)

    Since `1-FPR = F̂_source(t)` (source ECDF):

    $$
    T = \int TPR\cdot(1-FPR)^2\,dFPR = \int TPR\cdot\hat{F}_{\text{source}}(t)^2\,dFPR
    $$

    Uniform weight = AUC. Harm peaks at low `FPR` and decays to 0 at `FPR=1`. Target that clears a high bar source stays below gets a large `T`. `O(n log n)`.

### Visual intuition

```mermaid
xychart-beta
    title "Harmful vs beneficial — same AUC, different harm"
    x-axis "FPR →" [0, 0.2, 0.4, 0.6, 0.8, 1]
    y-axis "TPR ↑" 0 --> 1
    line "harmful (early rise)" [0, 0.75, 0.88, 0.94, 0.98, 1]
    line "beneficial (late rise)" [0, 0.08, 0.15, 0.30, 0.65, 1]
    line "diagonal" [0, 0.2, 0.4, 0.6, 0.8, 1]
```

*Harmful ROC jumps early (left, weighted) → harm LARGE. Beneficial rises late (right) → harm small but AUC still large.*

A shift toward better outcomes inflates AUC but not `T` — that's why the test is directional.

## AUC vs harm

|  | `test_shift` (AUC) | `test_harmful_shift` (harm) |
|---|---|---|
| Question | Did anything change? | Did target shift toward worse outcomes? |
| Statistic | `∫ TPR dFPR` | `∫ TPR·(1-FPR)² dFPR` |
| Weight | uniform | favours low `FPR` |
| Alternative | two-sided | one-sided `greater` |

Use `test_shift` when direction is unknown, `test_harmful_shift` once you can declare `worse`. Scores and weights stay fixed — only labels are permuted; see [API](../api/testing.md). For AUC `0.5` is chance; for harm compare observed to `result.null_distribution` median.

## References

- Kamulete, V. M. (2022). Test for non-negligible adverse shifts. *Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI)*. [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
