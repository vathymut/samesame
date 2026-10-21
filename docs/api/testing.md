# Testing API

These permutation tests compare one score per observation between two groups:
a reference (`source`) and a group under review (`target`). `test_shift` asks
whether their score distributions differ. `test_harm` tests for movement
toward a harmful tail declared with `worse` before the test.

For when to use each test, see [Benign vs harmful shift](../explanation/benign-vs-harmful-shift.md). For the weighted-AUC statistic, see [How the harm test works](../explanation/harmful-shift-statistic.md).

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
