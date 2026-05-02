# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Tests for samesame.weights."""

import numpy as np
import pytest

from samesame.weights import contextual_weights


# ---------------------------------------------------------------------------
# Numerical correctness — source mode
# ---------------------------------------------------------------------------


def test_source_mode_reweights_source_only(membership_probs):
    """In 'source' mode, target samples stay at weight 1."""
    result = contextual_weights(**membership_probs, mode="source")
    # source samples [0.25, 0.4] get RIW weights; target samples stay at 1.0
    assert np.allclose(result[2:], [1.0, 1.0])
    assert not np.allclose(result[:2], [1.0, 1.0])


def test_source_mode_balanced_default(membership_probs):
    """Default alpha_blend=0.5 with balanced groups gives expected source weights."""
    result = contextual_weights(**membership_probs, mode="source")
    expected = np.array([0.5, 0.8, 1.0, 1.0])
    assert np.allclose(result, expected)


def test_target_mode_reweights_target_only(membership_probs):
    """In 'target' mode, source samples stay at weight 1."""
    result = contextual_weights(**membership_probs, mode="target")
    expected = np.array([1.0, 1.0, 0.8, 0.5])
    assert np.allclose(result, expected)


def test_both_mode_reweights_all(membership_probs):
    """In 'both' mode, all samples are reweighted."""
    result = contextual_weights(**membership_probs, mode="both")
    expected = np.array([0.5, 0.8, 0.8, 0.5])
    assert np.allclose(result, expected)


def test_alpha_blend_one_gives_uniform(membership_probs):
    """alpha_blend=1.0 collapses to uniform weights for source mode."""
    result = contextual_weights(**membership_probs, mode="source", alpha_blend=1.0)
    assert np.allclose(result, [1.0, 1.0, 1.0, 1.0])


def test_alpha_blend_zero_gives_plain_density_ratio(membership_probs):
    """alpha_blend=0.0 gives plain density-ratio weights for source samples."""
    result = contextual_weights(**membership_probs, mode="source", alpha_blend=0.0)
    # source: r = p/(1-p), target: weight 1
    expected_source = np.array([0.25 / 0.75, 0.4 / 0.6])
    assert np.allclose(result[:2], expected_source)
    assert np.allclose(result[2:], [1.0, 1.0])


# ---------------------------------------------------------------------------
# balance parameter
# ---------------------------------------------------------------------------


def test_balance_true_and_false_agree_on_balanced_groups(membership_probs):
    """balance=True and balance=False agree when groups are already balanced."""
    result_true = contextual_weights(**membership_probs, mode="source", balance=True)
    result_false = contextual_weights(**membership_probs, mode="source", balance=False)
    # fixture has 2 source and 2 target → inferred ratio = 1.0 = explicit 1.0
    assert np.allclose(result_true, result_false)


def test_balance_true_vs_false_differ_on_unbalanced_groups():
    """balance=True and balance=False produce different results on unbalanced groups."""
    group = np.array([0, 0, 0, 1], dtype=int)
    membership_prob = np.array([0.4, 0.5, 0.6, 0.7])
    result_balanced = contextual_weights(
        group, membership_prob, mode="source", balance=True
    )
    result_unbalanced = contextual_weights(
        group, membership_prob, mode="source", balance=False
    )
    assert not np.allclose(result_balanced, result_unbalanced)


def test_balance_false_applies_unit_ratio():
    """balance=False treats groups as equal-sized (prior ratio = 1.0)."""
    group = np.array([0, 0, 0, 1], dtype=int)
    membership_prob = np.array([0.4, 0.5, 0.6, 0.7])
    result = contextual_weights(
        group, membership_prob, mode="source", balance=False, alpha_blend=0.0
    )
    # With ratio=1.0 and alpha_blend=0.0: weight = p/(1-p)
    expected_source = membership_prob[:3] / (1 - membership_prob[:3])
    assert np.allclose(result[:3], expected_source)


# ---------------------------------------------------------------------------
# ValueError: invalid membership_prob
# ---------------------------------------------------------------------------


def test_invalid_membership_prob_below_zero():
    group = np.array([0, 0, 1, 1])
    membership_prob = np.array([-0.1, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="membership_prob must be probabilities"):
        contextual_weights(group, membership_prob)


def test_invalid_membership_prob_at_zero():
    group = np.array([0, 0, 1, 1])
    membership_prob = np.array([0.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="membership_prob must be probabilities"):
        contextual_weights(group, membership_prob)


def test_invalid_membership_prob_at_one():
    group = np.array([0, 0, 1, 1])
    membership_prob = np.array([0.5, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="membership_prob must be probabilities"):
        contextual_weights(group, membership_prob)


# ---------------------------------------------------------------------------
# ValueError: invalid alpha_blend
# ---------------------------------------------------------------------------


def test_invalid_alpha_blend_too_low(membership_probs):
    with pytest.raises(ValueError, match="alpha_blend must be in"):
        contextual_weights(**membership_probs, alpha_blend=-0.1)


def test_invalid_alpha_blend_too_high(membership_probs):
    with pytest.raises(ValueError, match="alpha_blend must be in"):
        contextual_weights(**membership_probs, alpha_blend=1.1)


# ---------------------------------------------------------------------------
# ValueError: invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode_raises(membership_probs):
    with pytest.raises(ValueError, match="mode must be one of"):
        contextual_weights(**membership_probs, mode="not-a-mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ValueError: missing groups
# ---------------------------------------------------------------------------


def test_missing_source_group_raises():
    group = np.array([1, 1, 1, 1])
    membership_prob = np.array([0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        contextual_weights(group, membership_prob)


def test_length_mismatch_raises():
    group = np.array([0, 0, 1])
    membership_prob = np.array([0.2, 0.4, 0.6, 0.8])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        contextual_weights(group, membership_prob)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype(membership_probs):
    result = contextual_weights(**membership_probs)
    assert result.shape == (4,)
    assert result.dtype == np.float64
