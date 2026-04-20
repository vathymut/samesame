import numpy as np
import pytest


@pytest.fixture
def binary_scores(size: int = int(4e4)) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123_456)
    actual = rng.choice(2, size=size)
    predicted = rng.normal(size=size)
    return {"actual": actual, "predicted": predicted}


@pytest.fixture
def shift_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(321_654)
    reference = rng.normal(loc=0.0, scale=1.0, size=300)
    candidate = rng.normal(loc=0.35, scale=1.0, size=300)
    return {"reference": reference, "candidate": candidate}


@pytest.fixture
def binary_shift_samples() -> dict[str, np.ndarray]:
    reference = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=int)
    candidate = np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=int)
    return {"reference": reference, "candidate": candidate}


@pytest.fixture
def confidence_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(456_123)
    reference = rng.normal(loc=0.8, scale=0.08, size=250)
    candidate = rng.normal(loc=0.55, scale=0.08, size=250)
    return {"reference": reference, "candidate": candidate}


@pytest.fixture
def bayes_factors(size: int = 30) -> np.ndarray:
    rng = np.random.default_rng(123_456)
    return rng.uniform(low=0.1, high=10.0, size=size)


@pytest.fixture
def membership_probs() -> dict[str, np.ndarray]:
    actual = np.array([0, 0, 1, 1], dtype=int)
    predicted = np.array([0.25, 0.4, 0.6, 0.75])
    return {"actual": actual, "predicted": predicted}
