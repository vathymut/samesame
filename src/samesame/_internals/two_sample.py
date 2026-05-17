from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from samesame._internals.validation import as_numeric_vector


@dataclass(frozen=True)
class TwoSampleDataset:
    """Binary group labels and combined outlier scores for a two-sample test."""

    labels: NDArray[np.int_]
    scores: NDArray
    n_source: int
    n_target: int


def build_two_sample_dataset(source: ArrayLike, target: ArrayLike) -> TwoSampleDataset:
    """Build the aligned two-sample representation used by the testing seam."""
    source_scores = as_numeric_vector(source, name="source")
    target_scores = as_numeric_vector(target, name="target")
    labels = np.concatenate(
        (
            np.zeros(source_scores.shape[0], dtype=int),
            np.ones(target_scores.shape[0], dtype=int),
        )
    )
    scores = np.concatenate((source_scores, target_scores))
    return TwoSampleDataset(
        labels=labels,
        scores=scores,
        n_source=int(source_scores.shape[0]),
        n_target=int(target_scores.shape[0]),
    )


__all__ = ["TwoSampleDataset", "build_two_sample_dataset"]
