# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest

from samesame._data import concat_samples
from samesame._stats import EmpiricalPvalue
from samesame._wecdf import ECDFDiscrete, StepFunction


def test_concat_samples_2d_concatenates_on_rows():
    first = np.array([[1.0, 2.0], [3.0, 4.0]])
    second = np.array([[5.0, 6.0]])
    result = concat_samples([first, second])
    expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(result, expected)


def test_empirical_pvalue_greater_and_two_sided():
    observed = 0.5
    null_distribution = np.array([0.1, 0.5, 0.9, 1.2])
    pvalue = EmpiricalPvalue(observed=observed, null_distribution=null_distribution)

    greater = pvalue.greater()
    two_sided = pvalue.two_sided()

    assert np.isclose(greater, 4 / 5)
    assert np.isclose(two_sided, 1.0)


def test_step_function_invalid_side_raises():
    with pytest.raises(ValueError, match="side can take the values"):
        StepFunction([1, 2], [10, 20], side="middle")


def test_step_function_shape_mismatch_raises():
    with pytest.raises(ValueError, match="do not have the same shape"):
        StepFunction([1, 2], [10])


def test_step_function_non_1d_raises():
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        StepFunction(np.array([[1, 2]]), np.array([[10, 20]]))


def test_step_function_unsorted_input_is_sorted_and_callable():
    fn = StepFunction([2.0, 1.0], [20.0, 10.0], sorted=False, side="left")
    np.testing.assert_array_equal(fn.x, np.array([-np.inf, 1.0, 2.0]))
    np.testing.assert_array_equal(fn.y, np.array([0.0, 10.0, 20.0]))
    assert fn(1.5) == 10.0


def test_ecdf_discrete_without_freq_weights():
    ecdf = ECDFDiscrete(np.array([3, 1, 1, 2]))
    assert np.isclose(ecdf(1), 0.5)
    assert np.isclose(ecdf(2), 0.75)
    assert np.isclose(ecdf(3), 1.0)


def test_ecdf_discrete_with_freq_weights():
    x = np.array([3.0, 1.0, 2.0])
    freq_weights = np.array([1.0, 2.0, 1.0])
    ecdf = ECDFDiscrete(x, freq_weights=freq_weights)
    assert np.isclose(ecdf(1.0), 0.5)
    assert np.isclose(ecdf(2.0), 0.75)
    assert np.isclose(ecdf(3.0), 1.0)
