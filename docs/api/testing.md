# Testing API

These permutation tests compare one score per observation between a reference
group (`source`) and a group under review (`target`). `test_shift` uses ROC AUC
to test for rank-based separation: do target scores tend to be higher or lower
than source scores? It can miss changes in distribution shape that do not
change this ranking pattern.
`test_harm` tests whether target moves toward a harmful tail declared with
`worse` before the test.

For guidance on when to use each test, see
[Benign vs harmful shift](../explanation/benign-vs-harmful-shift.md). For the
weighted-AUC statistic, see
[How the harm test works](../explanation/harmful-shift-statistic.md).

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
