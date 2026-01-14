# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Tests for out-of-distribution (OOD) detection functions.

Tests verify implementations of LogitGap and MaxLogit scoring functions
against the specifications in "Revisiting Logit Distributions for Reliable
Out-of-Distribution Detection" (arXiv:2510.20134v1).
"""

from __future__ import annotations

import numpy as np
import pytest

from samesame.ood import logit_gap, max_logit


@pytest.fixture
def basic_logits() -> np.ndarray:
    """Basic logits for standard tests."""
    return np.array([[5.0, 1.0, 0.5], [2.0, 2.1, 1.9]])


@pytest.fixture
def id_samples() -> np.ndarray:
    """Peaked distributions (ID-like)."""
    return np.array([[5.0, 0.5, 0.3], [4.8, 0.4, 0.2], [5.1, 0.6, 0.4]])


@pytest.fixture
def ood_samples() -> np.ndarray:
    """Flat distributions (OOD-like)."""
    return np.array([[2.0, 1.9, 1.8], [2.1, 2.0, 1.9], [1.9, 1.8, 1.7]])


class TestMaxLogit:
    """Tests for the MaxLogit scoring function."""

    def test_basic(self, basic_logits: np.ndarray) -> None:
        """Test basic functionality."""
        scores = max_logit(basic_logits)
        expected = np.array([5.0, 2.1])
        np.testing.assert_array_almost_equal(scores, expected)

    @pytest.mark.parametrize(
        "logits,expected",
        [
            (np.array([[3.5, 1.2, 0.8]]), 3.5),  # Single sample
            (
                np.array([[-1.0, -2.0, -0.5], [-5.0, -3.0, -4.0]]),
                [-0.5, -3.0],
            ),  # Negative
            (np.array([[2.0, 2.0, 2.0]]), 2.0),  # All equal
            (np.array([[1000.0, 999.0, 998.0]]), 1000.0),  # Large values
        ],
    )
    def test_various_inputs(self, logits: np.ndarray, expected) -> None:
        """Test max_logit with various input types."""
        scores = max_logit(logits)
        if np.isscalar(expected):
            np.testing.assert_almost_equal(scores[0], expected)
        else:
            np.testing.assert_array_almost_equal(scores, expected)

    def test_output_properties(self) -> None:
        """Test output dtype and shape."""
        logits = np.random.randn(5, 10).astype(np.float32)
        scores = max_logit(logits)
        assert scores.dtype == np.float32
        assert scores.shape == (5,)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            np.array([5.0, 1.0, 0.5]),  # 1D
            np.ones((2, 3, 4)),  # 3D
        ],
    )
    def test_invalid_shapes(self, invalid_input: np.ndarray) -> None:
        """Test error handling for invalid shapes."""
        with pytest.raises(ValueError, match="must be 2D array"):
            max_logit(invalid_input)

    def test_single_class_error(self) -> None:
        """Test error for insufficient classes."""
        with pytest.raises(ValueError, match="at least 2 classes"):
            max_logit(np.array([[5.0], [3.0]]))

    def test_empty_classes_error(self) -> None:
        """Test error for zero classes."""
        with pytest.raises(ValueError, match="at least 2 classes"):
            max_logit(np.empty((2, 0), dtype=np.float32))


class TestLogitGap:
    """Tests for the LogitGap scoring function."""

    def test_basic(self, basic_logits: np.ndarray) -> None:
        """Test basic functionality with known values."""
        scores = logit_gap(basic_logits)
        # Sample 1: 5.0 - (1.0 + 0.5) / 2 = 4.25
        # Sample 2: 2.1 - (2.0 + 1.9) / 2 = 0.15
        np.testing.assert_array_almost_equal(scores, [4.25, 0.15])

    def test_invariance_to_sorting(self) -> None:
        """Test that input order doesn't affect output."""
        logits_unsorted = np.array([[1.0, 5.0, 0.5]])
        logits_sorted = np.array([[5.0, 1.0, 0.5]])
        np.testing.assert_almost_equal(
            logit_gap(logits_unsorted)[0], logit_gap(logits_sorted)[0]
        )

    def test_invariance_to_constant_shift(self) -> None:
        """Test that adding constant to all logits preserves gap."""
        logits1 = np.array([[5.0, 1.0, 0.5]])
        logits2 = np.array([[10.0, 6.0, 5.5]])
        np.testing.assert_almost_equal(logit_gap(logits1)[0], logit_gap(logits2)[0])

    @pytest.mark.parametrize(
        "logits,expected",
        [
            (np.array([[2.0, 2.0, 2.0]]), 0.0),  # Equal logits
            (np.array([[-1.0, -2.0, -0.5]]), -0.5 - (-1.0 + -2.0) / 2),  # Negative
        ],
    )
    def test_edge_cases(self, logits: np.ndarray, expected: float) -> None:
        """Test logit_gap with edge cases."""
        np.testing.assert_almost_equal(logit_gap(logits)[0], expected)

    def test_extreme_values(self) -> None:
        """Test numerical stability with extreme values."""
        logits_large = np.array([[1e6, 1e5, 1e4]])
        logits_small = np.array([[1e-6, 1e-7, 1e-8]])
        for logits in [logits_large, logits_small]:
            scores = logit_gap(logits)
            assert np.all(np.isfinite(scores))

    def test_output_properties(self) -> None:
        """Test output dtype and shape."""
        logits = np.random.randn(5, 10).astype(np.float32)
        scores = logit_gap(logits)
        assert scores.dtype == np.float32
        assert scores.shape == (5,)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            np.array([5.0, 1.0, 0.5]),  # 1D
            np.ones((2, 3, 4)),  # 3D
        ],
    )
    def test_invalid_shapes(self, invalid_input: np.ndarray) -> None:
        """Test error handling for invalid shapes."""
        with pytest.raises(ValueError, match="must be 2D array"):
            logit_gap(invalid_input)

    def test_insufficient_classes_error(self) -> None:
        """Test error for single class."""
        with pytest.raises(ValueError, match="at least 2 classes"):
            logit_gap(np.array([[5.0], [3.0]]))

    def test_empty_batch(self) -> None:
        """Test with empty batch."""
        logits = np.empty((0, 10), dtype=np.float32)
        scores = logit_gap(logits)
        assert scores.shape == (0,)

    def test_large_class_count(self) -> None:
        """Test with many classes."""
        logits = np.array([np.concatenate([[5.0], np.random.randn(999) * 0.1 + 0.5])])
        scores = logit_gap(logits)
        assert scores[0] > 4.0

    def test_type_conversion(self) -> None:
        """Test automatic type conversion to float32."""
        logits_int = np.array([[5, 1, 0]], dtype=np.int32)
        logits_float64 = np.array([[5.0, 1.0, 0.5]], dtype=np.float64)
        for logits in [logits_int, logits_float64]:
            scores = logit_gap(logits)
            assert scores.dtype == np.float32


class TestSeparationQuality:
    """Tests validating OOD detection separation quality."""

    def test_logit_gap_separates_id_ood(
        self, id_samples: np.ndarray, ood_samples: np.ndarray
    ) -> None:
        """Test that LogitGap provides ID-OOD separation."""
        id_scores = logit_gap(id_samples)
        ood_scores = logit_gap(ood_samples)
        assert np.mean(id_scores) > np.mean(ood_scores)

    def test_logit_gap_better_than_max_logit(
        self, id_samples: np.ndarray, ood_samples: np.ndarray
    ) -> None:
        """Test that LogitGap outperforms MaxLogit in separation."""
        gap_sep = np.mean(logit_gap(id_samples)) - np.mean(logit_gap(ood_samples))
        max_sep = np.mean(max_logit(id_samples)) - np.mean(max_logit(ood_samples))
        assert gap_sep > max_sep

    def test_same_max_different_gaps(self) -> None:
        """Test that LogitGap distinguishes distributions with same max."""
        logits_peaked = np.array([[5.0, 0.1, 0.0]])
        logits_flat = np.array([[5.0, 4.9, 4.8]])

        # MaxLogit cannot distinguish
        np.testing.assert_almost_equal(
            max_logit(logits_peaked)[0], max_logit(logits_flat)[0]
        )

        # LogitGap distinguishes
        gap_peaked = logit_gap(logits_peaked)[0]
        gap_flat = logit_gap(logits_flat)[0]
        assert gap_peaked > gap_flat


class TestNumericalStability:
    """Tests for numerical stability across input ranges."""

    @pytest.mark.parametrize(
        "logits",
        [
            np.array([[1e6, 1e5, 1e4]]),  # Very large
            np.array([[1e-6, 1e-7, 1e-8]]),  # Very small
            np.array([[5.0, -1.0, 2.0], [-3.0, 1.0, 0.0]]),  # Mixed signs
        ],
    )
    def test_all_functions_stable(self, logits: np.ndarray) -> None:
        """Test numerical stability for all functions."""
        for func in [max_logit, logit_gap]:
            scores = func(logits)
            assert np.all(np.isfinite(scores)), (
                f"{func.__name__} produced non-finite values"
            )

    def test_type_conversions(self) -> None:
        """Test automatic type conversions."""
        logits_int = np.array([[5, 1, 0], [2, 2, 1]], dtype=np.int32)
        logits_float64 = np.array([[5.0, 1.0, 0.5]], dtype=np.float64)

        for func in [max_logit, logit_gap]:
            scores_int = func(logits_int)
            scores_float64 = func(logits_float64)
            assert scores_int.dtype == np.float32
            assert scores_float64.dtype == np.float32
