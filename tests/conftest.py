import numpy as np
import pytest


@pytest.fixture
def shift_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(321_654)
    source = rng.normal(loc=0.0, scale=1.0, size=300)
    target = rng.normal(loc=0.35, scale=1.0, size=300)
    return {"source": source, "target": target}


@pytest.fixture
def confidence_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(456_123)
    source = rng.normal(loc=0.8, scale=0.08, size=250)
    target = rng.normal(loc=0.55, scale=0.08, size=250)
    return {"source": source, "target": target}


@pytest.fixture
def domain_probabilities() -> dict[str, np.ndarray]:
    return {
        "source": np.array([0.25, 0.4]),
        "target": np.array([0.6, 0.75]),
    }
