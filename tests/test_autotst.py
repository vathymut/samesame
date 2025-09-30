# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Test fixtures for different prediction scenarios are from:
https://github.com/jmkuebler/auto-tst/blob/main/tests/test_autotst.py
"""

from functools import partial

import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from samesame.ctst import CTST
from samesame.nit import WeightedAUC


@pytest.mark.slow
@pytest.mark.parametrize(
    "statistical_test",
    [
        WeightedAUC,
        partial(CTST, metric=roc_auc_score),
    ],
)
def test_scores(
    statistical_test,
    decent_predictions,
    somehow_undecided_predictions,
    undecided_predictions,
    very_undecided_predictions,
):
    inputs = [
        decent_predictions,
        somehow_undecided_predictions,
        undecided_predictions,
        very_undecided_predictions,
    ]
    ctsts = [statistical_test(**input) for input in inputs]
    pvals = [ctst.pvalue for ctst in ctsts]
    assert all(x <= y for x, y in zip(pvals, pvals[1:]))


def binarize(predictions):
    """Convert predictions to binary format."""
    return {
        "actual": predictions["actual"],
        "predicted": (predictions["predicted"] > 0.5).astype(int),
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "statistical_test",
    [
        partial(CTST, metric=balanced_accuracy_score),
        partial(CTST, metric=matthews_corrcoef),
    ],
)
def test_classes(
    statistical_test,
    decent_predictions,
    somehow_undecided_predictions,
    undecided_predictions,
    very_undecided_predictions,
):
    inputs = [
        decent_predictions,
        somehow_undecided_predictions,
        undecided_predictions,
        very_undecided_predictions,
    ]
    ctsts = [statistical_test(**binarize(input)) for input in inputs]
    pvals = [ctst.pvalue for ctst in ctsts]
    assert all(x <= y for x, y in zip(pvals, pvals[1:]))
