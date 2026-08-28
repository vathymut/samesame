# Common support

Region of feature space where source and target both have non-negligible density.

- **High overlap:** both groups plentiful → unweighted test is fine.
- **Low overlap:** points where the other group almost never goes (e.g., source has many 20-year-old students never seen in production). Those points can dominate an unweighted permutation test.

Weighting reframes the question:

- `reweight="source"` — down-weight source outliers.
- `reweight="target"` — down-weight target outliers.
- `reweight="both"` — mutual support.

Unweighted compares full populations; weighted compares overlap. If a strong unweighted signal vanishes after weighting, it was driven by low-overlap regions (see [Source reweighting](../../examples/weighting/source-reweighting.md)).

Check [ESS](ess.md) after weighting — if `ESS << n`, the weighted comparison rests on a few points.

See also: [Reweight](reweight.md), [Domain probability](domain-probability.md), [When importance weights help](../importance-weights-rationale.md).
