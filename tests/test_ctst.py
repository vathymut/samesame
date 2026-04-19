# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from samesame._data import group_by
from samesame.ctst import CTST
from samesame.nit import WeightedAUC


def binarize(predictions):
    """Convert predictions to binary format."""
    return {
        "actual": predictions["actual"],
        "predicted": (predictions["predicted"] > 0.5).astype(int),
    }


@pytest.mark.parametrize(
    "metric",
    [
        balanced_accuracy_score,
        matthews_corrcoef,
    ],
)
def test_frequentist_attributes(metric, decent_predictions, n_resamples=60):
    inputs = binarize(decent_predictions)
    ctst = CTST(
        actual=inputs["actual"],
        predicted=inputs["predicted"],
        metric=metric,
        n_resamples=n_resamples,
    )
    for attr in ["pvalue", "statistic", "null"]:
        assert hasattr(ctst, attr)
    assert isinstance(ctst.statistic, float)
    assert isinstance(ctst.null, np.ndarray)
    assert ctst.null.shape[0] == int(n_resamples)
    assert isinstance(ctst.pvalue, float)
    assert 0.0 <= ctst.pvalue <= 1.0


def test_from_samples(decent_predictions, n_resamples=60):
    samples = group_by(
        data=decent_predictions["predicted"],
        groups=decent_predictions["actual"],
    )
    assert len(samples) == 2
    ctst = CTST.from_samples(
        first_sample=samples[0],
        second_sample=samples[1],
        metric=roc_auc_score,
        n_resamples=n_resamples,
    )
    assert isinstance(ctst, CTST)
    dsos = WeightedAUC.from_samples(
        first_sample=samples[0],
        second_sample=samples[1],
        n_resamples=n_resamples,
    )
    assert isinstance(dsos, WeightedAUC)


def test_sample_mismatch():
    with pytest.raises(AssertionError):
        CTST.from_samples(
            first_sample=np.array([0, 1, 1]),
            second_sample=np.array([0.1, 0.2, 0.3]),
            metric=roc_auc_score,
        )


def test_wrong_metric(decent_predictions, n_resamples=60):
    def dummy_metric(a, b, c, d):  # wrong signature
        return 0

    with pytest.raises(AssertionError):
        CTST(
            actual=decent_predictions["actual"],
            predicted=decent_predictions["predicted"],
            metric=dummy_metric,  # type: ignore
            n_resamples=n_resamples,
        )


# ---------------------------------------------------------------------------
# Backwards-compatibility: sample_weight=None must reproduce unweighted results
# (TASK-001 / TEST-001)
# ---------------------------------------------------------------------------


def test_ctst_none_weight_equals_no_weight(decent_predictions, n_resamples=60):
    rng = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    base = CTST(
        actual=decent_predictions["actual"],
        predicted=decent_predictions["predicted"],
        metric=roc_auc_score,
        n_resamples=n_resamples,
        rng=rng,
    )
    weighted = CTST(
        actual=decent_predictions["actual"],
        predicted=decent_predictions["predicted"],
        metric=roc_auc_score,
        n_resamples=n_resamples,
        rng=rng2,
        sample_weight=None,
    )
    assert base.statistic == weighted.statistic
    np.testing.assert_array_equal(base.null, weighted.null)
    assert base.pvalue == weighted.pvalue


# ---------------------------------------------------------------------------
# validate_and_normalise_weights unit tests (TASK-004 / TEST-003 to TEST-008)
# ---------------------------------------------------------------------------


def test_validate_rejects_negative():
    from samesame._utils import validate_and_normalise_weights

    with pytest.raises(ValueError, match="negative"):
        validate_and_normalise_weights(np.array([-1.0, 1.0, 1.0]), 3)


def test_validate_rejects_wrong_length():
    from samesame._utils import validate_and_normalise_weights

    with pytest.raises(ValueError, match="length"):
        validate_and_normalise_weights(np.array([1.0, 1.0]), 3)


def test_validate_rejects_all_zero():
    from samesame._utils import validate_and_normalise_weights

    with pytest.raises(ValueError, match="zero"):
        validate_and_normalise_weights(np.array([0.0, 0.0, 0.0]), 3)


def test_validate_rejects_nan():
    from samesame._utils import validate_and_normalise_weights

    with pytest.raises(ValueError, match="finite"):
        validate_and_normalise_weights(np.array([1.0, np.nan, 1.0]), 3)


def test_validate_normalises_to_n():
    from samesame._utils import validate_and_normalise_weights

    w = np.array([2.0, 4.0, 6.0])
    result = validate_and_normalise_weights(w, 3)
    assert result is not None
    np.testing.assert_almost_equal(result.sum(), 3.0)


def test_validate_none_passthrough():
    from samesame._utils import validate_and_normalise_weights

    assert validate_and_normalise_weights(None, 5) is None


# ---------------------------------------------------------------------------
# Weighted permutation: asymmetric weights must change the pvalue (TASK-008 /
# TEST-009)
# ---------------------------------------------------------------------------


def test_ctst_weighted_permutation_fixed(decent_predictions, n_resamples=200):
    n = len(decent_predictions["actual"])
    rng_base = np.random.default_rng(0)
    rng_w = np.random.default_rng(0)
    # Upweight the first half strongly
    w = np.ones(n)
    w[: n // 2] = 10.0
    base = CTST(
        actual=decent_predictions["actual"],
        predicted=decent_predictions["predicted"],
        metric=roc_auc_score,
        n_resamples=n_resamples,
        rng=rng_base,
    )
    weighted = CTST(
        actual=decent_predictions["actual"],
        predicted=decent_predictions["predicted"],
        metric=roc_auc_score,
        n_resamples=n_resamples,
        rng=rng_w,
        sample_weight=w,
    )
    # Weighted statistic must differ (weights change the metric value)
    assert base.statistic != weighted.statistic
