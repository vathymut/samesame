---
applyTo: "src/**/*.py,tests/**/*.py"
description: "Performance guidance for numerical and statistical code in samesame"
---
# Performance guidance

Apply the repository-wide guidance from `../copilot-instructions.md` to any
hot-path or large-array work.

## Numerical performance

- Prefer vectorized NumPy, SciPy, and scikit-learn operations over Python loops
  for array-wide computations.
- Avoid repeated coercions, unnecessary intermediate arrays, and redundant
  passes over the same data in hot paths.
- Keep helpers side-effect free so they remain easy to benchmark, test, and
  reuse.

## Resampling and memory

- Treat permutation and weighting code as the main runtime-sensitive surface;
  validate size controls such as resample counts and batching before work
  begins.
- Consider both runtime and memory footprint when storing null distributions or
  copied arrays.
- Document trade-offs whenever an optimization changes readability, allocation
  patterns, or numerical stability.

## Practical validation

- Optimize only after identifying a real bottleneck or meaningful user cost.
- Preserve deterministic behavior when a caller provides a random seed or
  generator.
- Keep tests fast by using modest input sizes and focused assertions, even when
  production code supports larger workloads.