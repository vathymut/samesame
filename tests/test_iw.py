# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Tests for samesame.weights."""

import numpy as np
import pytest

from samesame.weights import from_domain_probabilities

# ---------------------------------------------------------------------------
# Numerical correctness — source mode
# ---------------------------------------------------------------------------


def test_source_mode_reweights_source_only(domain_probabilities):
    """In 'source' mode, target samples stay at weight 1."""
    result = from_domain_probabilities(**domain_probabilities, mode="source")
    # source samples [0.25, 0.4] get RIW weights; target samples stay at 1.0
    assert np.allclose(result.target, [1.0, 1.0])
    assert not np.allclose(result.source, [1.0, 1.0])


def test_source_mode_balanced_default(domain_probabilities):
    """Default lambda_=0.5 with balanced groups gives normalized source weights."""
    result = from_domain_probabilities(**domain_probabilities, mode="source")
    # raw RIW: [0.5, 0.8], sum=1.3; normalized to n_source=2: [10/13, 16/13]
    assert np.allclose(result.source, [10 / 13, 16 / 13])
    assert np.allclose(result.target, [1.0, 1.0])


def test_target_mode_reweights_target_only(domain_probabilities):
    """In 'target' mode, source samples stay at weight 1."""
    result = from_domain_probabilities(**domain_probabilities, mode="target")
    # raw RIW: [0.8, 0.5], sum=1.3; normalized to n_target=2: [16/13, 10/13]
    assert np.allclose(result.source, [1.0, 1.0])
    assert np.allclose(result.target, [16 / 13, 10 / 13])


def test_both_mode_reweights_all(domain_probabilities):
    """In 'both' mode, all samples are reweighted independently per group."""
    result = from_domain_probabilities(**domain_probabilities, mode="both")
    # source normalized: [10/13, 16/13]; target normalized: [16/13, 10/13]
    assert np.allclose(result.source, [10 / 13, 16 / 13])
    assert np.allclose(result.target, [16 / 13, 10 / 13])


def test_lambda_one_gives_uniform(domain_probabilities):
    """lambda_=1.0 collapses to uniform weights for source mode."""
    result = from_domain_probabilities(
        **domain_probabilities, mode="source", lambda_=1.0
    )
    assert np.allclose(result.source, [1.0, 1.0])
    assert np.allclose(result.target, [1.0, 1.0])


def test_lambda_zero_gives_plain_density_ratio(domain_probabilities):
    """lambda_=0.0 gives normalized plain density-ratio weights for source samples."""
    result = from_domain_probabilities(
        **domain_probabilities, mode="source", lambda_=0.0
    )
    # source: raw r = [1/3, 2/3], sum=1.0; normalized to n_source=2: [2/3, 4/3]
    expected_source = np.array([2 / 3, 4 / 3])
    assert np.allclose(result.source, expected_source)
    assert np.allclose(result.target, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Prior ratio inferred from group sizes
# ---------------------------------------------------------------------------


def test_prior_ratio_inferred_from_group_sizes():
    """Prior ratio n_src/n_tgt is inferred from array lengths; unequal sizes
    produce prior-corrected, normalized density-ratio weights."""
    # 3 source, 1 target → ratio = 3
    source_prob = np.array([0.4, 0.5, 0.6])
    target_prob = np.array([0.7])
    result = from_domain_probabilities(
        source_prob=source_prob, target_prob=target_prob, mode="source", lambda_=0.0
    )
    # raw density ratios: [2.0, 3.0, 4.5], sum=9.5; normalized to n_source=3
    expected_source = np.array([2.0, 3.0, 4.5]) * (3 / 9.5)
    assert np.allclose(result.source, expected_source)
    # target sample unaffected
    assert np.isclose(result.target[0], 1.0)


def test_equal_group_sizes_give_unit_prior_ratio(domain_probabilities):
    """Equal-sized groups infer a prior ratio of 1; normalized weights sum to n_source."""
    result = from_domain_probabilities(
        **domain_probabilities, mode="source", lambda_=0.0
    )
    # raw: [1/3, 2/3], sum=1.0; normalized to 2: result[0] = 2/3
    assert np.isclose(result.source[0], 2 / 3)
    assert np.isclose(result.source.sum(), 2.0)


# ---------------------------------------------------------------------------
# ValueError: invalid domain probabilities
# ---------------------------------------------------------------------------


def test_invalid_domain_prob_below_zero():
    with pytest.raises(ValueError, match="domain probabilities"):
        from_domain_probabilities(
            source_prob=np.array([-0.1, 0.5]), target_prob=np.array([0.5, 0.5])
        )


def test_invalid_domain_prob_at_zero():
    with pytest.raises(ValueError, match="domain probabilities"):
        from_domain_probabilities(
            source_prob=np.array([0.0, 0.5]), target_prob=np.array([0.5, 0.5])
        )


def test_invalid_domain_prob_at_one():
    with pytest.raises(ValueError, match="domain probabilities"):
        from_domain_probabilities(
            source_prob=np.array([0.5, 0.5]), target_prob=np.array([0.5, 1.0])
        )


# ---------------------------------------------------------------------------
# ValueError: invalid lambda_
# ---------------------------------------------------------------------------


def test_invalid_lambda_too_low(domain_probabilities):
    with pytest.raises(ValueError, match="lambda_ must be in"):
        from_domain_probabilities(**domain_probabilities, lambda_=-0.1)


def test_invalid_lambda_too_high(domain_probabilities):
    with pytest.raises(ValueError, match="lambda_ must be in"):
        from_domain_probabilities(**domain_probabilities, lambda_=1.1)


# ---------------------------------------------------------------------------
# ValueError: invalid mode
# ---------------------------------------------------------------------------


def test_invalid_mode_raises(domain_probabilities):
    with pytest.raises(ValueError, match="mode must be one of"):
        from_domain_probabilities(**domain_probabilities, mode="not-a-mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ValueError: empty arrays
# ---------------------------------------------------------------------------


def test_empty_source_prob_raises():
    with pytest.raises(ValueError, match="non-empty"):
        from_domain_probabilities(source_prob=np.array([]), target_prob=np.array([0.5]))


def test_empty_target_prob_raises():
    with pytest.raises(ValueError, match="non-empty"):
        from_domain_probabilities(source_prob=np.array([0.5]), target_prob=np.array([]))


# ---------------------------------------------------------------------------
# Normalization invariant — active group weights sum to sample size
# ---------------------------------------------------------------------------


def test_normalization_source_weights_sum_to_n_source(domain_probabilities):
    """Normalized source weights always sum to n_source regardless of lambda_."""
    n_source = len(domain_probabilities["source_prob"])
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = from_domain_probabilities(
            **domain_probabilities, mode="source", lambda_=lam
        )
        assert np.isclose(result.source.sum(), n_source), f"failed at lambda_={lam}"


def test_normalization_target_weights_sum_to_n_target(domain_probabilities):
    """Normalized target weights always sum to n_target regardless of lambda_."""
    n_target = len(domain_probabilities["target_prob"])
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = from_domain_probabilities(
            **domain_probabilities, mode="target", lambda_=lam
        )
        assert np.isclose(result.target.sum(), n_target), f"failed at lambda_={lam}"


def test_normalization_both_mode_independent(domain_probabilities):
    """In 'both' mode each group normalizes independently to its own sample size."""
    n_source = len(domain_probabilities["source_prob"])
    n_target = len(domain_probabilities["target_prob"])
    result = from_domain_probabilities(**domain_probabilities, mode="both")
    assert np.isclose(result.source.sum(), n_source)
    assert np.isclose(result.target.sum(), n_target)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_and_dtype(domain_probabilities):
    result = from_domain_probabilities(**domain_probabilities)
    assert result.source.shape == (2,)
    assert result.target.shape == (2,)
    assert result.source.dtype == np.float64
    assert result.target.dtype == np.float64
