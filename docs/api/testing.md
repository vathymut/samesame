# Testing API

Each test compares one score per observation across two samples: a reference
(`source`) and a sample under review (`target`). `test_shift` uses ROC AUC to
test rank-based separation: do target scores tend to be higher or lower
than source scores? It can miss changes in distribution shape that do not
change this ranking pattern. `test_harm` tests whether target moves toward the
score tail declared harmful with `worse` before testing.

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
