# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Set up fixtures for tests.

Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

from random import choice

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

_classifier_menu = (
    DummyClassifier,
    DecisionTreeClassifier,
    LogisticRegression,
    MLPClassifier,
    KNeighborsClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)

_regressor_menu = (
    DummyRegressor,
    DecisionTreeRegressor,
    LinearRegression,
    MLPRegressor,
    KNeighborsRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

_scorer_menu = (
    IsolationForest,
    KernelDensity,
)

rng = np.random.default_rng(123_456)


@pytest.fixture
def random_classifier():
    cls = choice(_classifier_menu)
    return cls()


@pytest.fixture
def random_regressor():
    reg = choice(_regressor_menu)
    return reg()


@pytest.fixture
def random_scorer():
    scorer = choice(_scorer_menu)
    return scorer()


@pytest.fixture
def data_classify(n_samples=100, seed=123_456):
    X, y = make_classification(
        n_samples=n_samples, class_sep=0.25, flip_y=0.25, random_state=seed
    )
    # returns tuple (X_train, X_test, y_train, y_test)
    return train_test_split(X, y, test_size=0.5)


@pytest.fixture
def data_regress(n_samples=100, seed=123_456):
    X, y, _ = make_regression(n_samples=n_samples, random_state=seed, coef=True)
    # returns tuple (X_train, X_test, y_train, y_test)
    return train_test_split(X, y, test_size=0.5)


@pytest.fixture
def binary_scores(size=int(4e4)) -> dict[str, np.ndarray]:
    actual = rng.choice(2, size=size)
    predicted = rng.normal(size=size)
    return {"actual": actual, "predicted": predicted}


@pytest.fixture
def perfect_predictions():
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    return {"actual": labels, "predicted": labels.copy()}


@pytest.fixture
def decent_predictions():
    preds1 = np.random.normal(0.75, 0.2, size=100)
    preds2 = np.random.normal(0.25, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    return {"actual": labels, "predicted": preds}


@pytest.fixture
def somehow_undecided_predictions():
    preds1 = np.random.normal(0.6, 0.2, size=100)
    preds2 = np.random.normal(0.4, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    return {"actual": labels, "predicted": preds}


@pytest.fixture
def undecided_predictions():
    preds1 = np.random.normal(0.55, 0.2, size=100)
    preds2 = np.random.normal(0.45, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    return {"actual": labels, "predicted": preds}


@pytest.fixture
def very_undecided_predictions():
    preds1 = np.random.normal(0.51, 0.2, size=100)
    preds2 = np.random.normal(0.49, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    return {"actual": labels, "predicted": preds}


@pytest.fixture
def unequal_length_predictions():
    preds = np.array([1.0] * 20)
    labels = np.array([1.0] * 10 + [0.0] * 11)
    return {"actual": labels, "predicted": preds}


@pytest.fixture
def wrong_shape_predictions():
    preds = np.random.normal([1, 3], [0.5, 1.5], size=(500, 2))
    labels = np.array([1.0] * 500 + [0.0] * 500)
    return {"actual": labels, "predicted": preds}


# %%
@pytest.fixture
def oob_rf_classifier(n_estimators=200, min_samples_leaf=10):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        oob_score=True,
        class_weight="balanced_subsample",
        random_state=123_456,
        verbose=1,
        n_jobs=-1,
    )


@pytest.fixture
def bayes_factors(size=30):
    return rng.uniform(low=0.1, high=10.0, size=size)


@pytest.fixture
def fixtures_for_importances():
    X, y, _ = make_regression(n_samples=100, n_features=5, random_state=42, coef=True)
    predictor = RandomForestRegressor(random_state=42).fit(X, y)
    imputer = SimpleImputer(strategy="mean").fit(X)
    return X, predictor, imputer
