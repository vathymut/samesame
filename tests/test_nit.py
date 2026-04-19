# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import pytest

from samesame.nit import WeightedAUC


def test_input_shape_and_length(
    unequal_length_predictions,
    wrong_shape_predictions,
):
    with pytest.raises(ValueError):
        WeightedAUC(**wrong_shape_predictions)
    with pytest.raises(ValueError):
        WeightedAUC(**unequal_length_predictions)


def test_bayesian_attributes(decent_predictions, n_resamples=60):
    ctst = WeightedAUC(**decent_predictions, n_resamples=n_resamples)
    for attr in ["posterior", "bayes_factor"]:
        assert hasattr(ctst, attr)
    assert isinstance(ctst.posterior, np.ndarray)
    assert ctst.posterior.shape[0] == int(n_resamples)
    assert ctst.bayes_factor > 0.0


# ---------------------------------------------------------------------------
# Backwards-compatibility: sample_weight=None must reproduce unweighted results
# (TASK-002 / TEST-002)
# ---------------------------------------------------------------------------


def test_wauc_none_weight_equals_no_weight(decent_predictions, n_resamples=60):
    rng = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    base = WeightedAUC(**decent_predictions, n_resamples=n_resamples, rng=rng)
    weighted = WeightedAUC(
        **decent_predictions, n_resamples=n_resamples, rng=rng2, sample_weight=None
    )
    assert base.statistic == weighted.statistic
    np.testing.assert_array_equal(base.null, weighted.null)
    assert base.pvalue == weighted.pvalue
    np.testing.assert_array_equal(base.posterior, weighted.posterior)
    assert base.bayes_factor == weighted.bayes_factor


# ---------------------------------------------------------------------------
# Weighted posterior: asymmetric weights must change posterior + bayes_factor
# (TASK-013 / TEST-010)
# ---------------------------------------------------------------------------


def test_wauc_weighted_posterior(decent_predictions, n_resamples=200):
    n = len(decent_predictions["actual"])
    rng_base = np.random.default_rng(1)
    rng_w = np.random.default_rng(1)
    w = np.ones(n)
    w[: n // 2] = 10.0
    base = WeightedAUC(**decent_predictions, n_resamples=n_resamples, rng=rng_base)
    weighted = WeightedAUC(
        **decent_predictions, n_resamples=n_resamples, rng=rng_w, sample_weight=w
    )
    # Posteriors must differ when weights are asymmetric
    assert not np.array_equal(base.posterior, weighted.posterior)
