"""Curated private implementation surface shared by public modules."""

from .bayes import _bayes_factor, as_bf, as_pvalue, bayes_factor, bayesian_posterior
from .permutation import run_permutation_test
from .statistics import get_shift_statistic, requires_binary_scores, wauc
from .two_sample import TwoSampleDataset, build_two_sample_dataset
from .validation import (
    Direction,
    RandomState,
    as_numeric_vector,
    resolve_random_state,
    validate_and_normalise_weights,
    validate_binary_actual_with_predicted,
    validate_direction,
)
from .weighting import WeightingMode, density_ratio, inverse_riw, riw, validate_mode

__all__ = [
    "Direction",
    "RandomState",
    "TwoSampleDataset",
    "WeightingMode",
    "_bayes_factor",
    "as_bf",
    "as_numeric_vector",
    "as_pvalue",
    "bayes_factor",
    "bayesian_posterior",
    "build_two_sample_dataset",
    "density_ratio",
    "get_shift_statistic",
    "inverse_riw",
    "requires_binary_scores",
    "resolve_random_state",
    "riw",
    "run_permutation_test",
    "validate_and_normalise_weights",
    "validate_binary_actual_with_predicted",
    "validate_direction",
    "validate_mode",
    "wauc",
]
