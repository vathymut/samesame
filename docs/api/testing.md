# Testing API

Permutation tests for one score per observation. `test_shift` asks whether source and target differ. `test_harm` asks whether target moved toward a harmful tail you declare before testing.

Which question to ask: [Benign vs harmful shift](../explanation/benign-vs-harmful-shift.md). The weighted AUC: [How the harm test works](../explanation/harmful-shift-statistic.md).

## Public functions

::: samesame.shift.test_shift
    options:
      show_root_heading: true
      heading_level: 3

::: samesame.shift.test_harm
    options:
      show_root_heading: true
      heading_level: 3

## Public result classes

::: samesame.shift.ShiftResult
    options:
      show_root_heading: true
      heading_level: 3

::: samesame.shift.HarmfulShiftResult
    options:
      show_root_heading: true
      heading_level: 3

## Public enum

::: samesame.shift.Worse
    options:
      show_root_heading: true
      heading_level: 3
