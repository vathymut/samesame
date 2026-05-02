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
    """lambda_=1.0 collapses to uniform weights for source mode."""
    result = contextual_weights(**membership_probs, mode="source", lambda_=1.0)
    assert np.allclose(result, [1.0, 1.0, 1.0, 1.0])


def test_alpha_blend_zero_gives_plain_density_ratio(membership_probs):
    """lambda_=0.0 gives plain density-ratio weights for source samples."""
    result = contextual_weights(**membership_probs, mode="source", lambda_=0.0)
    # source: r = p/(1-p), target: weight 1
    expected_source = np.array([0.25 / 0.75, 0.4 / 0.6])
    assert np.allclose(result[:2], expected_source)
    assert np.allclose(result[2:], [1.0, 1.0])


# ---------------------------------------------------------------------------
# Prior ratio inferred from group sizes
# ---------------------------------------------------------------------------


def test_prior_ratio_inferred_from_group_sizes():
    """Prior ratio n_src/n_tgt is inferred from array lengths; unequal sizes
    produce prior-corrected density-ratio weights."""
    # 3 source, 1 target → ratio = 3
    source_prob = np.array([0.4, 0.5, 0.6])
    target_prob = np.array([0.7])
    result = contextual_weights(
        source_prob=source_prob, target_prob=target_prob, mode="source", lambda_=0.0
    )
    # density ratio for source[0]=0.4 with ratio=3: r = 0.4/0.6 * 3 = 2.0
    assert np.isclose(result[0], 2.0)
    # target sample unaffected
    assert np.isclose(result[3], 1.0)


def test_equal_group_sizes_give_unit_prior_ratio(membership_probs):
    """Equal-sized groups infer a prior ratio of 1, matching explicit lambda_=0 behaviour."""
    result = contextual_weights(**membership_probs, mode="source", lambda_=0.0)
    # ratio=1, so density ratio for 0.25 is 0.25/0.75 = 1/3
    assert np.isclose(result[0], 0.25 / 0.75)


# ---------------------------------------------------------------------------
# ValueError: invalid membership_prob
# ---------------------------------------------------------------------------


def test_invalid_membership_prob_below_zero():
    with pytest.raises(ValueError, match="membership probabilities"):
        contextual_weights(
            source_prob=np.array([-0.1, 0.5]), target_prob=np.array([0.5, 0.5])
        )


def test_invalid_membership_prob_at_zero():
    with pytest.raises(ValueError, match="membership probabilities"):
        contextual_weights(
            source_prob=np.array([0.0, 0.5]), target_prob=np.array([0.5, 0.5])
        )


def test_invalid_membership_prob_at_one():
    with pytest.raises(ValueError, match="membership probabilities"):
        contextual_weights(
            source_prob=np.array([0.5, 0.5]), target_prob=np.array([0.5, 1.0])
        )


# ---------------------------------------------------------------------------
# ValueError: invalid alpha_blend
# ---------------------------------------------------------------------------


def test_invalid_alpha_blend_too_low(membership_probs):
    with pytest.raises(ValueError, match="lambda_ must be in"):
        contextual_weights(**membership_probs, lambda_=-0.1)


def test_invalid_alpha_blend_too_high(membership_probs):
    with pytest.raises(ValueError, match="lambda_ must be in"):
        contextual_weights(**membership_probs, lambda_=1.1)


# ---------------------------------------------------------------------------
# ValueError: invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode_raises(membership_probs):
    with pytest.raises(ValueError, match="mode must be one of"):
        contextual_weights(**membership_probs, mode="not-a-mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ValueError: empty arrays
# ---------------------------------------------------------------------------


def test_empty_source_prob_raises():
    with pytest.raises(ValueError, match="non-empty"):
        contextual_weights(source_prob=np.array([]), target_prob=np.array([0.5]))


def test_empty_target_prob_raises():
    with pytest.raises(ValueError, match="non-empty"):
        contextual_weights(source_prob=np.array([0.5]), target_prob=np.array([]))


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype(membership_probs):
    result = contextual_weights(**membership_probs)
    assert result.shape == (4,)
    assert result.dtype == np.float64
