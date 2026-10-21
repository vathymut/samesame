# Weighting API

Reweight a source-target comparison toward common support: the part of feature space where observations from both groups can occur. Weights change which observations count more; they cannot create overlap that is absent from the data.

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
