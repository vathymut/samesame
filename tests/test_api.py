from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

import samesame
from samesame import (
    AdverseShiftDetails,
    BayesianEvidence,
    ShiftDetails,
    adverse_shift_posterior,
    test_adverse_shift as run_adverse_shift_test,
    test_shift as run_shift_test,
)
from samesame.weights import ContextualWeights, contextual_weights


def test_root_exports() -> None:
    assert hasattr(samesame, "test_shift")
    assert hasattr(samesame, "test_adverse_shift")
    assert hasattr(samesame, "adverse_shift_posterior")
    assert hasattr(samesame, "weights")
    assert not hasattr(samesame, "advanced")
    assert not hasattr(samesame, "CTST")
    assert not hasattr(samesame, "DSOS")
    assert not hasattr(samesame, "ctst")
    assert not hasattr(samesame, "nit")


def test_signatures_are_keyword_only() -> None:
    shift_sig = inspect.signature(run_shift_test)
    adverse_sig = inspect.signature(run_adverse_shift_test)
    assert shift_sig.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert adverse_sig.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY


def test_test_shift_returns_shift_details(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples, n_resamples=64)
    assert isinstance(result, ShiftDetails)
    assert result.statistic_name == "roc_auc"
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.null_distribution.shape == (64,)


def test_test_shift_requires_keyword_arguments(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_shift_test(shift_samples["source"], shift_samples["target"])


def test_test_shift_rejects_unknown_statistic(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="statistic must be one of"):
        run_shift_test(**shift_samples, statistic="f1")  # type: ignore[arg-type]


def test_binary_only_statistics_require_binary_scores(
    shift_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="requires binary outlier scores"):
        run_shift_test(**shift_samples, statistic="balanced_accuracy")


@pytest.mark.parametrize("statistic", ["balanced_accuracy", "matthews_corrcoef"])
def test_binary_only_statistics_accept_binary_scores(
    binary_shift_samples: dict[str, np.ndarray],
    statistic: str,
) -> None:
    result = run_shift_test(**binary_shift_samples, statistic=statistic)  # type: ignore[arg-type]
    assert isinstance(result, ShiftDetails)
    assert result.statistic_name == statistic


def test_results_are_frozen(shift_samples: dict[str, np.ndarray]) -> None:
    result = run_shift_test(**shift_samples, n_resamples=64)
    with pytest.raises(FrozenInstanceError):
        result.pvalue = 0.0  # type: ignore[misc]


def test_test_adverse_shift_requires_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError):
        run_adverse_shift_test(**confidence_samples)


def test_test_adverse_shift_rejects_unknown_direction(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="direction must be one of"):
        run_adverse_shift_test(
            **confidence_samples,
            direction="up-is-bad",  # type: ignore[arg-type]
        )


def test_test_adverse_shift_handles_higher_is_better(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    primary = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
    )
    mirrored = run_adverse_shift_test(
        source=-confidence_samples["source"],
        target=-confidence_samples["target"],
        direction="higher-is-worse",
        n_resamples=64,
    )
    assert isinstance(primary, AdverseShiftDetails)
    assert primary.direction == "higher-is-better"
    assert np.isclose(primary.statistic, mirrored.statistic)
    assert np.isclose(primary.pvalue, mirrored.pvalue)


def test_shift_null_distribution_matches_n_resamples(
    shift_samples: dict[str, np.ndarray],
) -> None:
    result = run_shift_test(**shift_samples, n_resamples=99)
    assert result.null_distribution.shape == (99,)


def test_shift_supports_explicit_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    source, target = shift_samples["source"], shift_samples["target"]
    sample_weight = ContextualWeights(
        source=np.linspace(1.0, 3.0, len(source)),
        target=np.linspace(1.0, 3.0, len(target)),
    )
    base = run_shift_test(**shift_samples, n_resamples=64)
    weighted = run_shift_test(**shift_samples, n_resamples=64, weights=sample_weight)
    assert isinstance(weighted, ShiftDetails)
    assert base.statistic != weighted.statistic


def test_shift_supports_contextual_weights(
    shift_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(42)
    source, target = shift_samples["source"], shift_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = contextual_weights(
        source_prob=source_prob, target_prob=target_prob, mode="source"
    )
    base = run_shift_test(**shift_samples, n_resamples=64)
    contextual = run_shift_test(**shift_samples, n_resamples=64, weights=weights)
    assert isinstance(contextual, ShiftDetails)
    assert base.statistic != contextual.statistic


def test_adverse_shift_bayesian_evidence(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    result = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        rng=np.random.default_rng(0),
    )
    evidence = adverse_shift_posterior(
        **confidence_samples,
        direction="higher-is-better",
        result=result,
        n_resamples=64,
        rng=np.random.default_rng(42),
    )
    assert isinstance(result, AdverseShiftDetails)
    assert isinstance(evidence, BayesianEvidence)
    assert evidence.posterior.shape == (64,)
    assert isinstance(evidence.bayes_factor, float)


def test_adverse_shift_supports_contextual_weights(
    confidence_samples: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(99)
    source, target = confidence_samples["source"], confidence_samples["target"]
    source_prob = rng.uniform(0.2, 0.5, size=len(source))
    target_prob = rng.uniform(0.5, 0.8, size=len(target))
    weights = contextual_weights(
        source_prob=source_prob, target_prob=target_prob, mode="target"
    )
    base = run_adverse_shift_test(
        **confidence_samples, direction="higher-is-better", n_resamples=64
    )
    contextual = run_adverse_shift_test(
        **confidence_samples,
        direction="higher-is-better",
        n_resamples=64,
        weights=weights,
    )
    assert isinstance(contextual, AdverseShiftDetails)
    assert base.statistic != contextual.statistic
