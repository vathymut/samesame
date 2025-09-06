# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from scipy.stats import ranksums

from samesame._bayesboot import bayesian_posterior
from samesame.metrics import wauc


def test_wauc(binary_scores):
    actual = binary_scores["actual"]
    predicted = binary_scores["predicted"]
    result = wauc(actual, predicted)
    assert isinstance(result, float)
    assert np.allclose(result, 1 / 12, atol=0.01)


def test_posterior(binary_scores, n_resamples=100):
    actual = binary_scores["actual"]
    predicted = binary_scores["predicted"]
    result = bayesian_posterior(
        actual,
        predicted,
        metric=wauc,
        n_resamples=n_resamples,
    )
    assert np.allclose(result, 1, atol=1.0)
    assert result.shape == (n_resamples,)


def test_reproducibility(binary_scores, n_resamples=100):
    actual = binary_scores["actual"]
    predicted = binary_scores["predicted"]
    result1 = bayesian_posterior(
        actual,
        predicted,
        metric=wauc,
        n_resamples=n_resamples,
        rng=np.random.default_rng(42),
        n_jobs=1,
    )
    result2 = bayesian_posterior(
        actual,
        predicted,
        metric=wauc,
        n_resamples=n_resamples,
        rng=np.random.default_rng(42),
        n_jobs=1,
    )
    np.testing.assert_almost_equal(result1, result2)


def test_parallel(binary_scores, n_resamples=100):
    actual = binary_scores["actual"]
    predicted = binary_scores["predicted"]
    result_single = bayesian_posterior(
        actual,
        predicted,
        metric=wauc,
        n_resamples=n_resamples,
        rng=np.random.default_rng(42),
        n_jobs=1,
    )
    result_parallel = bayesian_posterior(
        actual,
        predicted,
        metric=wauc,
        n_resamples=n_resamples,
        rng=np.random.default_rng(42),
        n_jobs=2,
    )
    assert ranksums(result_single, result_parallel).pvalue > 0.05
