# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Tests for samesame.iw (AIWERM and RIWERM importance weighting)."""

import numpy as np
import pytest

from samesame.iw import aiw, riw


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_aiw_lam1_balanced(membership_probs):
    """aiw at lam=1.0 with balanced prior returns plain density ratios."""
    result = aiw(**membership_probs, lam=1.0)
    expected = [1 / 3, 2 / 3, 3 / 2, 3]
    assert np.allclose(result, expected)


def test_aiw_lam0_uniform(membership_probs):
    """aiw at lam=0.0 returns uniform weights regardless of predicted."""
    result = aiw(**membership_probs, lam=0.0)
    assert np.allclose(result, [1.0, 1.0, 1.0, 1.0])


def test_riw_lam0_equals_aiw_lam1(membership_probs):
    """riw at lam=0.0 is algebraically identical to aiw at lam=1.0 (plain IWERM)."""
    result_riw = riw(**membership_probs, lam=0.0)
    result_aiw = aiw(**membership_probs, lam=1.0)
    assert np.allclose(result_riw, result_aiw)


def test_riw_lam1_uniform(membership_probs):
    """riw at lam=1.0 returns uniform weights (r / r = 1)."""
    result = riw(**membership_probs, lam=1.0)
    assert np.allclose(result, [1.0, 1.0, 1.0, 1.0])


def test_riw_lam_half_balanced(membership_probs):
    """riw at default lam=0.5 with balanced prior returns expected values."""
    result = riw(**membership_probs, lam=0.5)
    expected = [0.5, 0.8, 1.2, 1.5]
    assert np.allclose(result, expected)


# ---------------------------------------------------------------------------
# ValueError: invalid predicted
# ---------------------------------------------------------------------------


def test_aiw_invalid_predicted_below_zero():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([-0.1, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="predicted must be membership probabilities"):
        aiw(actual, predicted)


def test_aiw_invalid_predicted_at_zero():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([0.0, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="predicted must be membership probabilities"):
        aiw(actual, predicted)


def test_aiw_invalid_predicted_at_one():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([0.5, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="predicted must be membership probabilities"):
        aiw(actual, predicted)


def test_riw_invalid_predicted_out_of_range():
    actual = np.array([0, 0, 1, 1])
    predicted = np.array([-0.1, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="predicted must be membership probabilities"):
        riw(actual, predicted)


# ---------------------------------------------------------------------------
# ValueError: invalid lam
# ---------------------------------------------------------------------------


def test_aiw_invalid_lam(membership_probs):
    with pytest.raises(ValueError, match="lam must be in"):
        aiw(**membership_probs, lam=-0.1)
    with pytest.raises(ValueError, match="lam must be in"):
        aiw(**membership_probs, lam=1.1)


def test_riw_invalid_lam(membership_probs):
    with pytest.raises(ValueError, match="lam must be in"):
        riw(**membership_probs, lam=-0.1)
    with pytest.raises(ValueError, match="lam must be in"):
        riw(**membership_probs, lam=1.1)


# ---------------------------------------------------------------------------
# prior_ratio inference and override
# ---------------------------------------------------------------------------


def test_inferred_prior_ratio_unbalanced():
    """Auto-inferred prior_ratio=3.0 from [0, 0, 0, 1] is applied correctly."""
    actual = np.array([0, 0, 0, 1])
    predicted = np.array([0.4, 0.5, 0.6, 0.7])
    result = aiw(actual, predicted, lam=1.0)
    # n_tr=3, n_te=1 -> prior_ratio=3.0
    expected = np.power((predicted / (1 - predicted)) * 3.0, 1.0)
    assert np.allclose(result, expected)


def test_prior_ratio_override():
    """Explicit prior_ratio overrides auto-inference."""
    actual = np.array([0, 0, 0, 1])
    predicted = np.array([0.4, 0.5, 0.6, 0.7])
    result_auto = aiw(actual, predicted, lam=1.0)
    result_override = aiw(actual, predicted, lam=1.0, prior_ratio=1.0)
    assert not np.allclose(result_auto, result_override)


def test_length_mismatch_raises():
    actual = np.array([0, 0, 1])
    predicted = np.array([0.2, 0.4, 0.6, 0.8])
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        aiw(actual, predicted)
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        riw(actual, predicted)


def test_missing_group_raises():
    actual = np.array([0, 0, 0, 0])
    predicted = np.array([0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError, match="both 0 and 1"):
        aiw(actual, predicted)
    with pytest.raises(ValueError, match="both 0 and 1"):
        riw(actual, predicted)


@pytest.mark.parametrize("bad_prior", [0.0, -1.0, np.inf, np.nan])
def test_invalid_prior_ratio_raises(membership_probs, bad_prior):
    with pytest.raises(ValueError, match="prior_ratio"):
        aiw(**membership_probs, prior_ratio=bad_prior)
    with pytest.raises(ValueError, match="prior_ratio"):
        riw(**membership_probs, prior_ratio=bad_prior)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype(membership_probs):
    """aiw and riw return float64 arrays with the same shape as predicted."""
    aiw_result = aiw(**membership_probs)
    riw_result = riw(**membership_probs)
    assert aiw_result.shape == (4,)
    assert aiw_result.dtype == np.float64
    assert riw_result.shape == (4,)
    assert riw_result.dtype == np.float64
