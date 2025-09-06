# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest

from samesame.bayes import as_bf, as_pvalue, bayes_factor


def test_conversion(bayes_factors):
    pvalues = as_pvalue(bayes_factors)
    round_trip = as_bf(pvalues)
    assert np.allclose(bayes_factors, round_trip)


def test_as_bf_scalar():
    """Test as_bf with a scalar p-value."""
    pvalue = 0.5
    expected_bf = 1.0
    result = as_bf(pvalue)
    assert np.isclose(result, expected_bf), f"Expected {expected_bf}, got {result}"


def test_as_bf_array():
    """Test as_bf with an array of p-values."""
    pvalues = np.array([0.05, 0.1, 0.5])
    expected_bfs = np.array([19.0, 9.0, 1.0])
    result = as_bf(pvalues)
    assert np.allclose(result, expected_bfs), f"Expected {expected_bfs}, got {result}"


def test_as_bf_edge_cases():
    """Test as_bf with edge case p-values."""
    pvalues = np.array([1e-10, 1 - 1e-10])
    result = as_bf(pvalues)
    assert np.isfinite(result).all(), "Bayes factor should be finite for valid p-values"


def test_as_bf_large_array():
    """Test as_bf with a large array of p-values."""
    pvalues = np.linspace(0.01, 0.99, 1000)
    result = as_bf(pvalues)
    assert np.all(result > 0), "All Bayes factors should be positive"


def test_as_pvalue_scalar():
    """Test as_pvalue with a scalar Bayes factor."""
    bayes_factor = 1.0
    expected_pvalue = 0.5
    result = as_pvalue(bayes_factor)
    assert np.isclose(
        result, expected_pvalue
    ), f"Expected {expected_pvalue}, got {result}"


def test_as_pvalue_array():
    """Test as_pvalue with an array of Bayes factors."""
    bayes_factors = np.array([19.0, 9.0, 1.0])
    expected_pvalues = np.array([0.05, 0.1, 0.5])
    result = as_pvalue(bayes_factors)
    assert np.allclose(
        result, expected_pvalues
    ), f"Expected {expected_pvalues}, got {result}"


def test_as_pvalue_edge_cases():
    """Test as_pvalue with edge case Bayes factors."""
    bayes_factors = np.array([1e-10, 1e10])
    result = as_pvalue(bayes_factors)
    assert np.isfinite(
        result
    ).all(), "P-values should be finite for valid Bayes factors"
    assert (np.array(result) > 0).all() and (
        np.array(result) < 1
    ).all(), "P-values should be in the range (0, 1)"


def test_as_pvalue_large_array():
    """Test as_pvalue with a large array of Bayes factors."""
    bayes_factors = np.logspace(-2, 2, 1000)
    result = as_pvalue(bayes_factors)
    assert (np.array(result) > 0).all() and (
        np.array(result) < 1
    ).all(), "All p-values should be in the range (0, 1)"


def test_as_pvalue_invalid_bf():
    """Test as_bf with invalid bayes factors."""
    invalid_bfs = [-0.1, -10.0]
    error_msg = "bayes_factor must be strictly positive"
    for bf in invalid_bfs:
        with pytest.raises(ValueError, match=error_msg):
            as_pvalue(bf)
    with pytest.raises(ValueError, match=error_msg):
        as_pvalue(np.array(invalid_bfs))


def test_bayes_factor_scalar():
    """Test bayes_factor with a small array of posterior samples."""
    posterior = np.array([0.2, 0.5, 0.8, 0.9])
    threshold = 0.5
    expected_bf = 1.0
    result = bayes_factor(posterior, threshold)
    assert np.isclose(result, expected_bf), f"Expected {expected_bf}, got {result}"


def test_bayes_factor_edge_cases():
    """Test bayes_factor with edge case posterior samples."""
    posterior = np.array([0.0, 0.0, 0.0, 0.0])
    threshold = 0.5
    result = bayes_factor(posterior, threshold)
    assert (
        result == 0.0
    ), "Bayes factor should be 0 when no samples exceed the threshold"

    posterior = np.array([1.0, 1.0, 1.0, 1.0])
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        result = bayes_factor(posterior, threshold)
        assert np.isinf(
            result
        ), "Bayes factor should be infinite when all samples exceed the threshold"


def test_bayes_factor_large_array():
    """Test bayes_factor with a large array of posterior samples."""
    posterior = np.random.uniform(0, 1, size=1000)
    threshold = 0.5
    result = bayes_factor(posterior, threshold)
    assert result > 0, "Bayes factor should be positive"
    assert np.isfinite(result), "Bayes factor should be finite for valid inputs"


def test_as_bf_invalid_pvalues():
    """Test as_bf with invalid p-values."""
    invalid_pvalues = [1.0, 0.0, -0.1, 1.1]
    error_msg = "pvalue must be within the open interval"
    for pvalue in invalid_pvalues:
        with pytest.raises(ValueError, match=error_msg):
            as_bf(pvalue)
    with pytest.raises(ValueError, match=error_msg):
        as_bf(np.array(invalid_pvalues))


def test_as_bf_clipped_pvalues():
    """Test as_bf with p-values near the boundaries."""
    pvalues = np.array([1e-12, 1 - 1e-12])
    result = as_bf(pvalues)
    assert np.isfinite(
        result
    ).all(), "Bayes factor should be finite for clipped p-values"
