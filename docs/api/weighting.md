# Weighting API

Use `source` for the reference sample and `target` for the sample under review.
Common support is the part of feature space where both groups have observations.
Reweighting changes how much each observation counts. It cannot create
overlap the data do not contain.

Worked example: [Weight for common support](../examples/credit/weight-for-common-support.md).

## Public functions

::: samesame.weights.domain_weights
    options:
      show_root_heading: true
      heading_level: 3

## Public data classes

::: samesame.weights.ImportanceWeights
    options:
      show_root_heading: true
      heading_level: 3

::: samesame.weights.EffectiveSampleSize
    options:
      show_root_heading: true
      heading_level: 3

## Public enum

::: samesame.weights.ReweightMode
    options:
      show_root_heading: true
      heading_level: 3
