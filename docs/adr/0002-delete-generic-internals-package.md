# Delete the generic _internals package

The generic `_internals` package is removed rather than kept as a private compatibility layer. Public concepts such as `Direction` and `WeightingMode` move to their owning public seams, while one-caller helpers such as result types, validation, two-sample assembly, WAUC support, the shift-statistic registry, permutation execution, and Bayesian evidence logic move into their owning public modules so ownership, change, and tests stay local instead of being split across shallow private files.

This cleanup prefers one larger owning module over immediately re-splitting private helpers by file. A larger `shift.py` or `weights.py` is acceptable until a second real caller or a clearly separate concept appears.

Public concepts stay with their owning seams rather than moving into a new shared `types` module. `Direction` belongs to the `shift` seam and `WeightingMode` belongs to the `weights` seam.

The root package surface stays namespace-first. Shared concepts and result types are not re-exported from `samesame` just because their ownership is clarified.
