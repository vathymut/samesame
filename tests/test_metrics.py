# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Harmful-shift statistic behavior."""

from __future__ import annotations

import numpy as np
import pytest

from samesame._statistics import harmful_shift_statistic

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def separated() -> dict[str, np.ndarray]:
    """Perfectly separated source (negatives) and target (positives)."""
    rng = np.random.default_rng(11)
    actual = np.array([0] * 40 + [1] * 40)
    predicted = np.concatenate([rng.uniform(0.0, 0.3, 40), rng.uniform(0.7, 1.0, 40)])
    return {"actual": actual, "predicted": predicted}


@pytest.fixture
def mixed() -> dict[str, np.ndarray]:
    """Overlapping source and target — moderate discrimination."""
    rng = np.random.default_rng(22)
    actual = np.array([0] * 60 + [1] * 60)
    predicted = np.concatenate([rng.normal(0.4, 0.15, 60), rng.normal(0.6, 0.15, 60)])
    return {"actual": actual, "predicted": predicted}


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


def test_wauc_returns_float(mixed: dict[str, np.ndarray]) -> None:
    result = harmful_shift_statistic(**mixed)
    assert isinstance(result, float)


def test_wauc_result_bounded(mixed: dict[str, np.ndarray]) -> None:
    result = harmful_shift_statistic(**mixed)
    assert 0.0 <= result <= 1.0


def test_wauc_result_bounded_with_weights(mixed: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(33)
    weights = rng.uniform(0.5, 2.0, len(mixed["actual"]))
    result = harmful_shift_statistic(**mixed, sample_weight=weights)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Separation behaviour
# ---------------------------------------------------------------------------


def test_wauc_harmful_shift_exceeds_reverse_shift(
    separated: dict[str, np.ndarray],
) -> None:
    """WAUC is higher when target scores exceed source scores (harmful shift)
    than when source scores exceed target scores (reverse)."""
    result_harm = harmful_shift_statistic(**separated)
    result_reverse = harmful_shift_statistic(
        separated["actual"],
        1.0 - separated["predicted"],
    )
    assert result_harm > result_reverse


def test_wauc_sensitive_to_shift_direction(mixed: dict[str, np.ndarray]) -> None:
    """Harmfully shifted data (target higher) yields higher WAUC than reversed."""
    result_harm = harmful_shift_statistic(**mixed)
    result_reverse = harmful_shift_statistic(mixed["actual"], 1.0 - mixed["predicted"])
    assert result_harm > result_reverse


# ---------------------------------------------------------------------------
# Weight propagation
# ---------------------------------------------------------------------------


def test_wauc_uniform_weights_match_unweighted(mixed: dict[str, np.ndarray]) -> None:
    """Uniform sample_weight must produce the same result as no weights."""
    unweighted = harmful_shift_statistic(**mixed)
    uniform = np.ones(len(mixed["actual"]), dtype=float)
    weighted = harmful_shift_statistic(**mixed, sample_weight=uniform)
    assert np.isclose(unweighted, weighted)


def test_wauc_asymmetric_negative_weights_change_result(
    mixed: dict[str, np.ndarray],
) -> None:
    """Weights that reshape the negative-class ECDF must change the result."""
    actual = mixed["actual"]
    predicted = mixed["predicted"]
    n = len(actual)
    neg_idx = np.where(actual == 0)[0]
    neg_by_score = neg_idx[np.argsort(predicted[neg_idx])]
    half = len(neg_by_score) // 2
    # a: up-weight low-scoring negatives; b: up-weight high-scoring negatives
    weights_a = np.ones(n)
    weights_b = np.ones(n)
    weights_a[neg_by_score[:half]] = 5.0
    weights_b[neg_by_score[half:]] = 5.0
    result_a = harmful_shift_statistic(actual, predicted, sample_weight=weights_a)
    result_b = harmful_shift_statistic(actual, predicted, sample_weight=weights_b)
    assert not np.isclose(result_a, result_b)


def test_wauc_weights_affect_negative_class_ecdf(
    mixed: dict[str, np.ndarray],
) -> None:
    """Up-weighting hard negatives should reduce wauc relative to unweighted."""
    actual = mixed["actual"]
    predicted = mixed["predicted"]
    # Find the highest-scoring negative (hardest for the classifier)
    neg_idx = np.where(actual == 0)[0]
    hardest_neg = neg_idx[np.argmax(predicted[neg_idx])]
    # Up-weighting that sample should make the ECDF heavier at high thresholds,
    # changing the integration result.
    weights_base = np.ones(len(actual), dtype=float)
    weights_hard = weights_base.copy()
    weights_hard[hardest_neg] = 20.0
    result_base = harmful_shift_statistic(actual, predicted, sample_weight=weights_base)
    result_hard = harmful_shift_statistic(actual, predicted, sample_weight=weights_hard)
    assert not np.isclose(result_base, result_hard)


# ---------------------------------------------------------------------------
# Invalid weights
# ---------------------------------------------------------------------------


def test_wauc_rejects_all_zero_negative_class_weights(
    mixed: dict[str, np.ndarray],
) -> None:
    """All-zero weights for the negative class raise instead of returning NaN."""
    actual = mixed["actual"]
    weights = np.ones(len(actual), dtype=float)
    weights[actual == 0] = 0.0
    with pytest.raises(ValueError, match="freq_weights must not be all zero"):
        harmful_shift_statistic(actual, mixed["predicted"], sample_weight=weights)


def test_wauc_rejects_negative_freq_weights(mixed: dict[str, np.ndarray]) -> None:
    """Negative weights for the negative class raise explicitly."""
    actual = mixed["actual"]
    weights = np.ones(len(actual), dtype=float)
    weights[actual == 0] = -1.0
    with pytest.raises(ValueError, match="freq_weights must be non-negative"):
        harmful_shift_statistic(actual, mixed["predicted"], sample_weight=weights)


def test_weighted_ecdf_rejects_wrong_length_freq_weights() -> None:
    """The weighted ECDF raises when frequencies do not match the data length."""
    from samesame._statistics import _weighted_ecdf

    x = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="freq_weights must have the same length"):
        _weighted_ecdf(x, query=np.array([0.15]), freq_weights=np.ones(2))


def test_weighted_ecdf_rejects_non_finite_freq_weights() -> None:
    """The weighted ECDF raises on NaN or inf weights instead of returning NaN."""
    from samesame._statistics import _weighted_ecdf

    x = np.array([0.1, 0.2, 0.3])
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="freq_weights must contain only finite"):
            _weighted_ecdf(
                x,
                query=np.array([0.15]),
                freq_weights=np.array([bad, 1.0, 1.0]),
            )
