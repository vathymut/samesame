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
