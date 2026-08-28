# Permutation test

Label-permutation null used by `test_shift` and `test_harmful_shift`.

- **Null:** exchangeability — source and target come from the same distribution, so labels carry no information.
- **Procedure:** scores (and weights, if given) stay fixed; only labels are permuted `n_resamples` times, recomputing the statistic each time.
- **p-value:** fraction of null statistics at least as extreme as observed, with +1 smoothing (Phipson & Smyth) in `(0, 1]`.

Specifics:

- `test_shift` — two-sided on ROC AUC; doubling capped at `1`.
- `test_harmful_shift` — one-sided `greater` on the harm statistic `∫ TPR·(1-FPR)² dFPR`.

??? example "Simplified implementation — `src/samesame/_permutation.py`"
    ```python
    sample_weight = np.concatenate((weights.source, weights.target)) if weights else None
    observed = metric(labels, scores, sample_weight)
    for i in range(n_resamples):
        perm_labels = rng.permutation(labels)
        null[i] = metric(perm_labels, scores, sample_weight)
    ```

`n_resamples` — permutation resamples. Default `9999`. Use `999` while exploring, `9999` for final, `19999` if you need `p < 0.001` resolution. Cost `O(n log n)` per resample, `O(n)` memory.

**Tip:** compare `result.statistic` to `result.null_distribution`. If observed lies beyond `quantile(null, 0.95)`, `p < 0.05`. For AUC `0.5` is chance; for harm compare to null median — no fixed `0.5`.

References: Phipson & Smyth (2010), Kamulete et al. (2022) [arXiv:2107.02990](https://arxiv.org/abs/2107.02990).
